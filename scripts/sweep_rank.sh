#!/usr/bin/env bash
# LoRA rank sweep -> the quality-vs-trainable-parameters chart in the README.
# ~4 sequential runs; start it before you go to bed on a Mac.
set -euo pipefail
CONFIG="${1:-configs/qwen3_1p7b_mps.yaml}"

for R in 8 16 32 64; do
  echo "=== LoRA r=$R ==="
  python -m src.train --config "$CONFIG" --lora-r "$R" --output "runs/r${R}"
  python -m src.evaluate --config "$CONFIG" --adapter "runs/r${R}/final" --tag "lora_r${R}"
done

python scripts/make_report.py
