"""Device, dtype, attention and quantization resolution.

This is the module that makes one config file run unchanged on an Apple
Silicon laptop and on an A100. Every backend-specific branch in the repo
should live here rather than being scattered through the training code.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
from dataclasses import asdict, dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class Backend:
    device: str  # "cuda" | "mps" | "cpu"
    dtype: torch.dtype  # compute dtype for model weights
    attn_impl: str  # "flash_attention_2" | "sdpa" | "eager"
    supports_4bit: bool  # bitsandbytes availability
    pin_memory: bool
    num_workers: int
    bf16: bool  # pass to TrainingArguments
    fp16: bool  # pass to TrainingArguments

    def summary(self) -> str:
        d = asdict(self)
        d["dtype"] = str(self.dtype)
        return " | ".join(f"{k}={v}" for k, v in d.items())


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _mps_bf16_ok() -> bool:
    """bf16 on MPS is reliable on M2 and later, flaky on M1.

    torch has no direct chip query, so we read the CPU brand string. When we
    cannot identify the chip we take the conservative branch (fp32), because a
    silently-wrong dtype produces NaN losses that are painful to debug.
    """
    if platform.system() != "Darwin":
        return False
    try:
        import subprocess

        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - best effort only
        return False
    # "Apple M1", "Apple M1 Pro", "Apple M2 Max", "Apple M3", "Apple M4" ...
    for gen in ("M2", "M3", "M4", "M5"):
        if gen in brand:
            return True
    return False


def resolve_backend(
    prefer_4bit: bool = False,
    force_device: str | None = None,
    force_attn: str | None = None,
    allow_mps_bf16: bool | None = None,
) -> Backend:
    """Pick a coherent set of backend settings.

    Args:
        prefer_4bit: request QLoRA. Honoured only on CUDA; bitsandbytes has no
            MPS backend, so on a Mac this downgrades to bf16/fp32 LoRA with a
            warning rather than crashing deep inside the model load.
        force_device: override autodetection ("cuda"/"mps"/"cpu").
        force_attn: override attention implementation.
        allow_mps_bf16: override the M1/M2 heuristic. None means autodetect.
    """
    if force_device:
        device = force_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    supports_4bit = False
    bf16 = fp16 = False

    if device == "cuda":
        cuda_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if cuda_bf16 else torch.float16
        bf16, fp16 = cuda_bf16, not cuda_bf16
        # flash-attn-2 needs Ampere or newer in addition to the package.
        major = torch.cuda.get_device_capability()[0]
        if _has_package("flash_attn") and major >= 8:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        supports_4bit = _has_package("bitsandbytes")
        if prefer_4bit and not supports_4bit:
            log.warning("4-bit requested but bitsandbytes is not installed; using %s", dtype)
        pin_memory = True
        num_workers = min(8, (os.cpu_count() or 2))

    elif device == "mps":
        use_bf16 = _mps_bf16_ok() if allow_mps_bf16 is None else allow_mps_bf16
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        # Do NOT set fp16=True here. HF's fp16 path builds a torch.cuda
        # GradScaler and will fail on MPS. bf16 needs no scaler.
        bf16, fp16 = use_bf16, False
        # flash-attn-2 is a CUDA extension. sdpa works on recent torch+MPS;
        # eager is the escape hatch when a model's sdpa path hits an
        # unimplemented op and silently falls back to CPU.
        attn_impl = "sdpa"
        if prefer_4bit:
            log.warning(
                "4-bit (QLoRA) requested but bitsandbytes has no MPS backend. "
                "Falling back to %s LoRA. Use a CUDA config for QLoRA.",
                dtype,
            )
        # MPS uses unified memory; pinning is meaningless and worker processes
        # add overhead without overlapping a real host->device copy.
        pin_memory = False
        num_workers = 0

    else:  # cpu
        dtype = torch.float32
        attn_impl = "eager"
        pin_memory = False
        num_workers = min(4, (os.cpu_count() or 2))

    if force_attn:
        attn_impl = force_attn

    return Backend(
        device=device,
        dtype=dtype,
        attn_impl=attn_impl,
        supports_4bit=supports_4bit and prefer_4bit,
        pin_memory=pin_memory,
        num_workers=num_workers,
        bf16=bf16,
        fp16=fp16,
    )


def set_seed_everywhere(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def peak_memory_gb(device: str) -> float | None:
    """Peak allocated memory, for the README table."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    if device == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1024**3
        except Exception:  # noqa: BLE001
            return None
    return None
