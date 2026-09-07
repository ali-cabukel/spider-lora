"""Generate predictions and score execution accuracy.

Zero-shot baseline (run this FIRST, before any training):
    python -m src.evaluate --config configs/qwen3_1p7b_mps.yaml --tag zeroshot

After fine-tuning:
    python -m src.evaluate --config configs/qwen3_1p7b_mps.yaml \
        --adapter runs/qwen3_r16/final --tag lora_r16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import GenerationCollator, Text2SQLDataset, load_spider
from .env import peak_memory_gb, resolve_backend, set_seed_everywhere
from .eval_exec import aggregate, score_one
from .modeling import load_for_inference, load_tokenizer
from .runlog import RunLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="omit for the zero-shot baseline")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--out", default="results")
    return ap.parse_args()


@torch.no_grad()
def generate_all(model, tokenizer, loader, backend, max_new_tokens: int) -> list[dict]:
    preds: list[dict] = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        meta = batch.pop("meta")
        batch = {k: v.to(backend.device) for k, v in batch.items()}
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: eval must be deterministic
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[:, batch["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for m, t in zip(meta, texts):
            preds.append({**m, "raw": t})
        if bi % 10 == 0:
            done = len(preds)
            log.info("generated %d (%.2f s/example)", done, (time.time() - t0) / max(done, 1))
    return preds


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed_everywhere(cfg.get("seed", 42))
    backend = resolve_backend(
        prefer_4bit=cfg["model"].get("load_in_4bit", False),
        force_device=args.device,
        force_attn=cfg["model"].get("attn_implementation"),
        allow_mps_bf16=cfg["model"].get("allow_mps_bf16"),
    )
    log.info("backend: %s | adapter: %s", backend.summary(), args.adapter or "none (zero-shot)")

    tokenizer = load_tokenizer(cfg["model"]["id"], padding_side="left")
    examples = load_spider(args.split, cfg["data"].get("spider_dir"), args.limit)
    ds = Text2SQLDataset(
        examples,
        tokenizer,
        db_root=cfg["data"]["db_root"],
        max_len=cfg["data"]["max_len"],
        sample_rows=cfg["data"].get("sample_rows", 0),
        train=False,
    )
    bs = args.batch_size or cfg.get("eval", {}).get("batch_size", 4)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=GenerationCollator(tokenizer.pad_token_id),
        num_workers=0,  # generation is GPU-bound; workers only add overhead
    )

    model = load_for_inference(cfg["model"]["id"], backend, args.adapter)

    t0 = time.time()
    preds = generate_all(model, tokenizer, loader, backend, cfg.get("eval", {}).get("max_new_tokens", 256))
    gen_seconds = time.time() - t0

    rows = [
        score_one(cfg["data"]["db_root"], p["db_id"], p["question"], p["gold"], p["raw"])
        for p in preds
    ]
    summary = aggregate(rows)
    summary.update(
        {
            "tag": args.tag,
            "adapter": args.adapter,
            "split": args.split,
            "backend": backend.summary(),
            "gen_seconds": gen_seconds,
            "sec_per_example": gen_seconds / max(len(preds), 1),
            "peak_memory_gb": peak_memory_gb(backend.device),
        }
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.tag}_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__) + "\n")

    logger = RunLogger(cfg.get("log_db", "runs/runs.sqlite"), config=summary)
    logger.log_predictions(rows)
    logger.finish(notes=f"eval:{args.tag}")

    print("\n" + "=" * 58)
    print(f"  {args.tag}")
    print("=" * 58)
    print(f"  execution accuracy : {summary['execution_accuracy']:.4f}")
    print(f"  correct / scorable : {summary['n_correct']} / {summary['n_scorable']}")
    print(f"  unscorable gold    : {summary['n_gold_broken']}")
    print(f"  sec / example      : {summary['sec_per_example']:.2f}")
    print("  errors:")
    for k, v in summary["error_taxonomy"].items():
        print(f"    {k:<20} {v}")
    print("=" * 58)


if __name__ == "__main__":
    main()
