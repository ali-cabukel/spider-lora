"""Offline tests for schema rendering and loss masking.

Uses a stub tokenizer so this runs without torch-heavy model downloads.
Requires torch (for the collator tensors); skips cleanly if absent.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP test_data.py: torch not installed")
    raise SystemExit(0)

from src.data import (  # noqa: E402
    IGNORE_INDEX,
    CausalCollator,
    Example,
    GenerationCollator,
    Text2SQLDataset,
    render_schema,
)


class StubTokenizer:
    """Whitespace tokenizer. Stands in for a real tokenizer's interface."""

    eos_token = "<eos>"
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}

    def _id(self, t: str) -> int:
        return self.vocab.setdefault(t, len(self.vocab))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(t) for t in text.split()]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        if "enable_thinking" not in kw:
            raise TypeError("stub requires enable_thinking to mimic Qwen3")
        body = " ".join(m["content"] for m in messages)
        return f"<|im_start|> {body} <|im_start|>assistant"


def make_db(root: Path, db_id="toy") -> None:
    d = root / db_id
    d.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(d / f"{db_id}.sqlite")
    con.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INT);
        CREATE TABLE song (id INTEGER PRIMARY KEY, singer_id INT REFERENCES singer(id), title TEXT);
        INSERT INTO singer VALUES (1,'Ada',34);
        INSERT INTO song VALUES (1,1,'Alpha');
        """
    )
    con.commit()
    con.close()


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_db(root)
        render_schema.cache_clear()

        schema = render_schema(str(root), "toy")
        ok &= check("schema lists both tables", "TABLE singer" in schema and "TABLE song" in schema)
        ok &= check("schema marks the primary key", "id INTEGER PK" in schema)
        ok &= check("schema records the foreign key", "FK song.singer_id -> singer.id" in schema)
        ok &= check("schema omits sample rows by default", "SAMPLE" not in schema)
        ok &= check(
            "sample_rows injects rows when asked",
            "SAMPLE" in render_schema(str(root), "toy", sample_rows=1),
        )

        render_schema.cache_clear()
        render_schema(str(root), "toy")
        render_schema(str(root), "toy")
        ok &= check("schema render is cached", render_schema.cache_info().hits >= 1)

        tok = StubTokenizer()
        exs = [Example("toy", "How old is Ada?", "SELECT age FROM singer WHERE name = 'Ada'")]

        # --- training mode: prompt masked, answer supervised ---------------- #
        ds = Text2SQLDataset(exs, tok, str(root), max_len=512, train=True)
        item = ds[0]
        n_sup = sum(1 for x in item["labels"] if x != IGNORE_INDEX)
        n_mask = sum(1 for x in item["labels"] if x == IGNORE_INDEX)
        ok &= check("labels align with input_ids", len(item["labels"]) == len(item["input_ids"]))
        ok &= check("prompt tokens are masked out", n_mask > 0)
        ok &= check("answer tokens are supervised", n_sup > 0)
        ok &= check(
            "supervised span is the tail of the sequence",
            item["labels"][-n_sup:] == item["input_ids"][-n_sup:],
        )
        ok &= check("mask precedes supervision", item["labels"][0] == IGNORE_INDEX)

        # --- truncation keeps the answer ----------------------------------- #
        short = Text2SQLDataset(exs, tok, str(root), max_len=8, train=True)
        t = short[0]
        ok &= check("truncates to max_len", len(t["input_ids"]) == 8)
        ok &= check(
            "left-truncation preserves supervised tokens (no NaN loss)",
            any(x != IGNORE_INDEX for x in t["labels"]),
        )
        ok &= check("truncation is counted", short.truncation_count == 1)

        # --- collators ------------------------------------------------------ #
        batch = CausalCollator(tok.pad_token_id)([ds[0], short[0]])
        ok &= check("train collator pads to a multiple of 8", batch["input_ids"].shape[1] % 8 == 0)
        ok &= check(
            "padding is masked in labels and attention",
            (batch["labels"][1][batch["attention_mask"][1] == 0] == IGNORE_INDEX).all().item(),
        )

        eval_ds = Text2SQLDataset(exs * 2, tok, str(root), max_len=512, train=False)
        e0 = eval_ds[0]
        ok &= check("eval items carry gold + meta", {"gold", "db_id", "question"} <= set(e0))
        ok &= check("eval items have no labels", "labels" not in e0)

        gbatch = GenerationCollator(tok.pad_token_id)([eval_ds[0], eval_ds[1]])
        ok &= check(
            "generation collator LEFT-pads (mask ends with 1s)",
            gbatch["attention_mask"][:, -1].all().item(),
        )

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
