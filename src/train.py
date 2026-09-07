"""LoRA fine-tuning entrypoint.

    python -m src.train --config configs/qwen3_1p7b_mps.yaml
    python -m src.train --config configs/qwen3_1p7b_cuda.yaml

The same config runs on both backends; everything device-specific is resolved
in src/env.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import CausalCollator, Text2SQLDataset, load_spider
from .env import empty_cache, peak_memory_gb, resolve_backend, set_seed_everywhere
from .modeling import attach_lora, load_model, load_tokenizer
from .runlog import RunLogger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="truncate train set (smoke test)")
    ap.add_argument("--lora-r", type=int, default=None, help="override for the rank sweep")
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return ap.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.limit is not None:
        cfg["data"]["train_limit"] = args.limit
    if args.output:
        cfg["output_dir"] = args.output
    return cfg


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float = 0.1):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, tokenizer, out_dir: Path, step: int, optimizer=None, scheduler=None):
    d = out_dir / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)  # adapter weights only, a few MB
    tokenizer.save_pretrained(d)
    if optimizer is not None:
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
            },
            d / "trainer_state.pt",
        )
    log.info("saved checkpoint -> %s", d)
    return d


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    set_seed_everywhere(cfg.get("seed", 42))
    backend = resolve_backend(
        prefer_4bit=cfg["model"].get("load_in_4bit", False),
        force_device=args.device,
        force_attn=cfg["model"].get("attn_implementation"),
        allow_mps_bf16=cfg["model"].get("allow_mps_bf16"),
    )
    log.info("backend: %s", backend.summary())

    out_dir = Path(cfg.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["model"]["id"], padding_side="right")

    train_examples = load_spider(
        "train", cfg["data"].get("spider_dir"), cfg["data"].get("train_limit")
    )
    train_ds = Text2SQLDataset(
        train_examples,
        tokenizer,
        db_root=cfg["data"]["db_root"],
        max_len=cfg["data"]["max_len"],
        sample_rows=cfg["data"].get("sample_rows", 0),
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        num_workers=backend.num_workers,
        pin_memory=backend.pin_memory,
        drop_last=True,
    )

    model = load_model(
        cfg["model"]["id"],
        backend,
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True),
        for_training=True,
    )
    model, n_trainable = attach_lora(
        model,
        r=cfg["lora"]["r"],
        alpha=cfg["lora"].get("alpha"),
        dropout=cfg["lora"].get("dropout", 0.05),
        target_modules=cfg["lora"].get("target_modules"),
    )

    accum = cfg["train"].get("grad_accum", 1)
    epochs = cfg["train"]["epochs"]
    steps_per_epoch = len(loader) // accum
    total_steps = steps_per_epoch * epochs

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["train"].get("weight_decay", 0.0)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, total_steps, cfg["train"].get("warmup_ratio", 0.03))

    start_step = 0
    if args.resume:
        state = torch.load(Path(args.resume) / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        log.info("resumed at step %d", start_step)

    logger = RunLogger(cfg.get("log_db", "runs/runs.sqlite"), config={**cfg, "backend": backend.summary()})
    log.info(
        "steps/epoch=%d total=%d trainable=%s", steps_per_epoch, total_steps, f"{n_trainable:,}"
    )

    model.train()
    step = start_step
    t0 = time.time()
    tokens_seen = 0
    running = 0.0

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(backend.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            running += loss.item()
            tokens_seen += int(batch["attention_mask"].sum().item())

            if (i + 1) % accum != 0:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["train"].get("max_grad_norm", 1.0),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg["train"].get("log_every", 10) == 0:
                elapsed = time.time() - t0
                tps = tokens_seen / max(elapsed, 1e-6)
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    "epoch %d step %d/%d loss %.4f lr %.2e |g| %.2f %.0f tok/s",
                    epoch,
                    step,
                    total_steps,
                    running / cfg["train"].get("log_every", 10),
                    lr_now,
                    float(gnorm),
                    tps,
                )
                logger.log(
                    step,
                    loss=running / cfg["train"].get("log_every", 10),
                    lr=lr_now,
                    grad_norm=float(gnorm),
                    tokens_per_sec=tps,
                )
                running = 0.0

            if cfg["train"].get("save_every") and step % cfg["train"]["save_every"] == 0:
                save_checkpoint(model, tokenizer, out_dir, step, optimizer, scheduler)
                empty_cache(backend.device)

    final = save_checkpoint(model, tokenizer, out_dir, step, optimizer, scheduler)
    (out_dir / "final").parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")

    peak = peak_memory_gb(backend.device)
    meta = {
        "backend": backend.summary(),
        "peak_memory_gb": peak,
        "total_steps": step,
        "wall_seconds": time.time() - t0,
        "tokens_per_sec": tokens_seen / max(time.time() - t0, 1e-6),
        "trainable_params": n_trainable,
        "truncated_examples": train_ds.truncation_count,
        "run_id": logger.run_id,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("done: %s", json.dumps(meta, indent=2))
    logger.finish(notes=f"final={final}")


if __name__ == "__main__":
    main()
