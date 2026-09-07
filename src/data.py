"""Spider loading, schema rendering, prompt construction and loss masking.

Design note: the table schema shown to the model is read directly out of the
SQLite files rather than from Spider's `tables.json`. The eval harness executes
against those same files, so this guarantees the schema the model sees and the
schema it is graded against can never drift apart.
"""

from __future__ import annotations

import functools
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch

log = logging.getLogger(__name__)

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a text-to-SQL model. Given a SQLite schema and a question, reply "
    "with a single executable SQLite query and nothing else. No explanation, "
    "no markdown fences."
)


@dataclass
class Example:
    db_id: str
    question: str
    query: str


def db_path(db_root: str | Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"


@functools.lru_cache(maxsize=512)
def render_schema(db_root: str, db_id: str, sample_rows: int = 0) -> str:
    """Serialize a SQLite database as a compact CREATE-TABLE-ish description.

    Cached because Spider reuses each database across many questions.
    """
    path = db_path(db_root, db_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing database {path}. Run `python scripts/get_spider.py` first."
        )
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    lines: list[str] = []
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]
        for table in tables:
            cur.execute(f'PRAGMA table_info("{table}")')
            cols = [f"{r[1]} {r[2] or 'TEXT'}" + (" PK" if r[5] else "") for r in cur.fetchall()]
            lines.append(f"TABLE {table}({', '.join(cols)})")
            cur.execute(f'PRAGMA foreign_key_list("{table}")')
            for fk in cur.fetchall():
                lines.append(f"  FK {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            if sample_rows:
                try:
                    cur.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}')
                    rows = cur.fetchall()
                    if rows:
                        lines.append(f"  SAMPLE {rows}")
                except sqlite3.Error:
                    pass
    finally:
        con.close()
    return "\n".join(lines)


def build_messages(ex: Example, schema: str) -> list[dict]:
    user = f"Schema:\n{schema}\n\nQuestion: {ex.question}\n\nSQL:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, ex: Example, schema: str) -> str:
    """Prompt string ending exactly where the model should start generating."""
    kwargs = {}
    # Qwen3 emits <think> blocks unless thinking is explicitly disabled. Other
    # templates do not accept this kwarg, so probe once and degrade quietly.
    try:
        return tokenizer.apply_chat_template(
            build_messages(ex, schema),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            build_messages(ex, schema),
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class Text2SQLDataset(torch.utils.data.Dataset):
    """Tokenized Spider examples with prompt tokens masked out of the loss.

    Masking matters: without it roughly 90% of the loss comes from reproducing
    the schema, and the model spends its capacity learning to copy the prompt.
    """

    def __init__(
        self,
        examples: list[Example],
        tokenizer,
        db_root: str,
        max_len: int = 1024,
        sample_rows: int = 0,
        train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train
        self.db_root = db_root
        self.sample_rows = sample_rows
        self.examples = examples
        self._n_truncated = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        schema = render_schema(self.db_root, ex.db_id, self.sample_rows)
        prompt = render_prompt(self.tokenizer, ex, schema)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not self.train:
            return {
                "input_ids": prompt_ids[-self.max_len :],
                "db_id": ex.db_id,
                "question": ex.question,
                "gold": ex.query,
            }

        answer = ex.query.strip() + self.tokenizer.eos_token
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_len:
            # Truncate from the LEFT so the answer and the end of the schema
            # survive. Right-truncation would cut off the target entirely and
            # produce an all-masked label row (NaN loss).
            self._n_truncated += 1
            input_ids = input_ids[-self.max_len :]
            labels = labels[-self.max_len :]

        return {"input_ids": input_ids, "labels": labels}

    @property
    def truncation_count(self) -> int:
        return self._n_truncated


@dataclass
class CausalCollator:
    """Right-pads for training. Generation uses a left-padding collator."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            maxlen = ((maxlen + m - 1) // m) * m

        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class GenerationCollator:
    """Left-pads so that every sequence ends at the same position."""

    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn = [], []
        for b in batch:
            ids = b["input_ids"]
            pad = maxlen - len(ids)
            input_ids.append([self.pad_token_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "meta": [
                {"db_id": b["db_id"], "question": b["question"], "gold": b["gold"]}
                for b in batch
            ],
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_spider(split: str, data_dir: str | None = None, limit: int | None = None) -> list[Example]:
    """Load Spider from local JSON if present, else from the HF hub.

    Local files are preferred so that a run is reproducible offline once the
    dataset has been fetched.
    """
    if data_dir:
        fname = {"train": "train_spider.json", "validation": "dev.json", "dev": "dev.json"}[split]
        path = Path(data_dir) / fname
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            rows = [Example(r["db_id"], r["question"], r["query"]) for r in raw]
            log.info("Loaded %d %s examples from %s", len(rows), split, path)
            return rows[:limit] if limit else rows

    from datasets import load_dataset

    hf_split = "validation" if split in ("validation", "dev") else "train"
    ds = load_dataset("xlangai/spider", split=hf_split)
    rows = [Example(r["db_id"], r["question"], r["query"]) for r in ds]
    log.info("Loaded %d %s examples from the HF hub", len(rows), split)
    return rows[:limit] if limit else rows
