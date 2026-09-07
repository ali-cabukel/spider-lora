# spider-lora

LoRA fine-tuning of a small instruct model for text-to-SQL, graded by
**execution accuracy** on Spider. One config file runs unchanged on an Apple
Silicon laptop (MPS) and on a CUDA GPU.

> **Status: baseline implemented, results not yet measured.** The tables below
> are intentionally empty. Fill them from `python scripts/make_report.py` after
> your own runs — do not publish numbers you have not reproduced.

## Results

### Execution accuracy (Spider dev, n=1034)

| run | exec acc | correct / n | sec/example | hardware |
|---|---|---|---|---|
| zero-shot | _TBD_ | | | |
| LoRA r=8 | _TBD_ | | | |
| LoRA r=16 | _TBD_ | | | |
| LoRA r=32 | _TBD_ | | | |
| LoRA r=64 | _TBD_ | | | |

### Error taxonomy

| run | wrong_result | bad_column | bad_table | syntax | timeout | empty |
|---|---|---|---|---|---|---|
| zero-shot | | | | | | |
| LoRA r=16 | | | | | | |

### Throughput

| backend | dtype | attn | tok/s (train) | peak mem | sec/example (eval) |
|---|---|---|---|---|---|
| M-series MPS | bf16 | sdpa | | | |
| A100 40GB | nf4 + bf16 | flash_attn_2 | | | |

---

## Why execution accuracy

`WHERE a > 5 AND b = 2` and `WHERE b = 2 AND a > 5` are the same query. Exact
string match calls one of them wrong. This harness runs both the gold and the
predicted query against the actual SQLite file and compares result sets:

- **Order-sensitive only when the gold query has an `ORDER BY`.** Otherwise rows
  are compared as a multiset.
- **Column order still matters** — right values under the wrong headers answered
  a different question.
- **Values are normalised** so `3` and `3.0` match, and floats are compared to
  6dp (a `SUM` should not fail on a last-bit difference).
- **Queries are interrupted after 10s** via a SQLite progress handler, so one
  runaway cartesian join cannot stall the eval loop.
- **Unscorable items are quarantined.** If the *gold* query fails to execute,
  the item is excluded from the denominator instead of silently counting as a
  model error.

Prediction cleanup (`extract_sql`) strips markdown fences, Qwen3 `<think>`
blocks and a leading `SQL:` label, then takes the first statement. It
deliberately does **not** repair broken SQL — that would inflate the score with
work the model did not do.

## Setup

```bash
git clone https://github.com/ali-cabukel/spider-lora.git && cd spider-lora

conda create -n spider-lora python=3.11 -y
conda activate spider-lora

# Install torch for YOUR platform first:
#   Mac:  pip install torch
#   CUDA: see https://pytorch.org/get-started/locally/
pip install -r requirements.txt

python scripts/get_spider.py    # fetches splits, verifies the sqlite databases
```

`scripts/get_spider.py` pulls the question/SQL pairs from `xlangai/spider` on
the HF hub, then downloads the SQLite databases from
`HAL-9001/spider-databases` (the Yale project page is 404; the official HF
dataset does not ship the `.sqlite` files). Execution accuracy cannot run
without those databases.

## Running

```bash
# 1. Zero-shot baseline FIRST. A fine-tune with no baseline proves nothing.
python -m src.evaluate --config configs/qwen3_1p7b_mps.yaml --tag zeroshot

# 2. Smoke test the training loop on 64 examples before committing hours
python -m src.train --config configs/qwen3_1p7b_mps.yaml --limit 64

# 3. Full run
python -m src.train --config configs/qwen3_1p7b_mps.yaml

# 4. Score it
python -m src.evaluate --config configs/qwen3_1p7b_mps.yaml \
    --adapter runs/qwen3_1p7b_r16/final --tag lora_r16

# 5. Rank sweep -> the quality-vs-parameters chart
bash scripts/sweep_rank.sh

# 6. Regenerate the tables above
python scripts/make_report.py
```

The identical commands work on CUDA with `configs/qwen3_1p7b_cuda.yaml`. The two
configs use the **same effective batch size (16) and learning rate**, so the
runs are directly comparable; only the batch/accum split and precision differ.

## Portability: what actually breaks on a Mac

Every backend branch lives in `src/env.py`. The rest of the codebase never
checks the device.

| Issue | Handling |
|---|---|
| **bitsandbytes has no MPS backend** | `load_in_4bit: true` is honoured on CUDA and downgraded to bf16 LoRA with a warning on MPS, instead of failing inside the model load. There is no QLoRA on Apple Silicon. |
| **HF `fp16=True` builds a CUDA GradScaler** | Never set on MPS. bf16 needs no scaler; the fp16 path is CUDA-only. |
| **bf16 is reliable on M2+, flaky on M1** | Autodetected from the CPU brand string, overridable via `allow_mps_bf16`. Unknown chips take the conservative fp32 branch, because a bad dtype shows up as a NaN loss hours later. |
| **flash-attn-2 is a CUDA extension needing SM80+** | `sdpa` on MPS, `flash_attention_2` on CUDA only when the package is present *and* compute capability ≥ 8. `eager` is the escape hatch when an sdpa op silently falls back to CPU. |
| **`pin_memory` is meaningless on unified memory** | Disabled on MPS along with dataloader workers, which add overhead without overlapping a real host→device copy. |

## Design notes

**Loss is computed on the completion only.** Prompt tokens are set to `-100`.
In a representative example the prompt is 56 tokens and the answer is 8 — without
masking, ~88% of the gradient signal is the model learning to copy the schema
back.

**Truncation is left-sided.** Right-truncating a long schema would cut off the
target entirely and produce an all-masked label row, i.e. a NaN loss. Truncated
examples are counted and reported in `train_meta.json`.

**The schema shown to the model is read from the SQLite files themselves**, not
from Spider's `tables.json`. Since eval executes against those same files, the
schema the model sees and the schema it is graded against cannot drift apart.

**`alpha` defaults to `2*r`.** Holding alpha fixed while sweeping rank changes
the effective learning rate, which would confound "more capacity" with "larger
LR" in the rank ablation.

**Generation is greedy and left-padded.** Sampling would make eval
non-reproducible; right-padding would leave sequences ending at different
positions.

**Logging goes to a local SQLite file** (`runs/runs.sqlite`), so anyone cloning
this can reproduce a run and plot curves without a W&B account.

## Tests

No GPU or model download required:

```bash
python tests/test_eval_exec.py   # 19 checks: comparison semantics, taxonomy, timeout
python tests/test_data.py        # schema rendering, masking, collators
```

`test_eval_exec.py` builds a throwaway SQLite database and asserts that
predicate reordering passes, `ORDER BY` direction is graded, `3 == 3.0`, column
swaps fail, each error class is correctly labelled, a recursive-CTE bomb is
interrupted, and unscorable gold is excluded from the denominator.

## Layout

```
src/env.py         backend resolution (all device branching lives here)
src/data.py        Spider loading, schema rendering, prompts, masking, collators
src/modeling.py    model + LoRA construction
src/train.py       training loop, cosine schedule, checkpoint/resume
src/evaluate.py    batched greedy generation + scoring
src/eval_exec.py   execution accuracy, error taxonomy
src/runlog.py      local SQLite run logger
scripts/           dataset fetch, rank sweep, report generation
tests/             offline tests, no GPU needed
```

## Known limitations

- Single-GPU only. No FSDP/DeepSpeed path.
- Schema is passed in full; no retrieval over large schemas, which is the main
  blocker for scaling this to BIRD-sized databases.
- Execution accuracy can reward a query that is right by coincidence on this
  particular data (e.g. a missing filter that happens to exclude nothing).
  Test-suite accuracy fixes this and is not implemented.
- No values-in-schema prompting by default (`sample_rows: 0`); enabling it
  usually helps and is an easy ablation.

## Next

- [ ] Zero-shot baseline number
- [ ] LoRA rank sweep 8/16/32/64
- [ ] Train on Spider, evaluate on BIRD — does it learn SQL or memorise Spider?
- [ ] Error taxonomy with 5–10 annotated failure cases
- [ ] Attention-only vs attention+MLP adapter ablation
