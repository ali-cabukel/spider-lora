"""Model + LoRA construction, parameterised by the resolved Backend."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import Backend

log = logging.getLogger(__name__)

# Llama / Qwen / Mistral family projection names. Attention-only adapters
# (drop the last three) train faster but consistently score lower on
# text-to-SQL, which is worth showing as an ablation rather than assuming.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_tokenizer(model_id: str, padding_side: str = "right"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        # Reusing EOS as PAD is safe here because the collator builds an
        # explicit attention mask and pads labels with -100.
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(
    model_id: str,
    backend: Backend,
    gradient_checkpointing: bool = True,
    for_training: bool = True,
):
    kwargs: dict = {
        "dtype": backend.dtype,
        "attn_implementation": backend.attn_impl,
        "trust_remote_code": True,
    }

    if backend.supports_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=backend.dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if not backend.supports_4bit:
        # device_map is deliberately unused off the quantized path: on MPS it
        # invokes accelerate's dispatch machinery for no benefit, and on a
        # single GPU a plain .to() is clearer.
        model = model.to(backend.device)

    if for_training:
        model.config.use_cache = False  # incompatible with grad checkpointing
        if backend.supports_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
        elif gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    else:
        model.config.use_cache = True
        model.eval()

    return model


def attach_lora(
    model,
    r: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        # alpha = 2r is a reasonable default; holding alpha fixed while sweeping
        # r conflates "more capacity" with "larger effective learning rate".
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "LoRA r=%d alpha=%s | trainable %s / %s (%.3f%%)",
        r,
        cfg.lora_alpha,
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return model, trainable


def load_for_inference(model_id: str, backend: Backend, adapter_path: str | None = None):
    """Base model, optionally with a trained adapter merged in."""
    model = load_model(model_id, backend, gradient_checkpointing=False, for_training=False)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if not backend.supports_4bit:
            # Merging removes the adapter's separate matmuls from the hot path.
            # Cannot be done against 4-bit base weights.
            model = model.merge_and_unload()
        model.eval()
    return model


@torch.no_grad()
def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
