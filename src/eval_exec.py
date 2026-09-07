"""Execution accuracy: run the predicted query and compare result sets.

Chosen over exact string match because there are many correct spellings of the
same query. A model that writes `WHERE a > 5 AND b = 2` instead of
`WHERE b = 2 AND a > 5` is not wrong, and string match says it is.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Rough guard against a runaway cartesian join eating the eval loop.
DEFAULT_TIMEOUT_S = 10.0
_PROGRESS_OPS = 10_000


class QueryTimeout(Exception):
    pass


# --------------------------------------------------------------------------- #
# Output cleaning
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of raw model output.

    Kept deliberately conservative: this cleans up formatting the model was
    told not to produce, it does not repair broken SQL. Repairing here would
    inflate the score with work the model did not do.
    """
    text = _THINK.sub("", text)
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # Drop a leading echo of the prompt marker if the template leaked one.
    text = re.sub(r"^(SQL|Answer|Query)\s*:\s*", "", text, flags=re.IGNORECASE)
    # Take the first statement only.
    if ";" in text:
        text = text.split(";", 1)[0]
    return " ".join(text.split()).strip()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    # Spider databases contain latin-1 and other non-UTF8 bytes; without this
    # a handful of otherwise-correct queries raise on decode.
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def execute(db_file: Path, query: str, timeout_s: float = DEFAULT_TIMEOUT_S):
    """Run a read-only query. Returns rows, or raises."""
    import time

    con = _connect(db_file)
    deadline = time.monotonic() + timeout_s

    def _handler():
        return 1 if time.monotonic() > deadline else 0

    try:
        con.set_progress_handler(_handler, _PROGRESS_OPS)
        cur = con.cursor()
        cur.execute(query)
        return cur.fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise QueryTimeout(str(e)) from e
        raise
    finally:
        con.set_progress_handler(None, 0)
        con.close()


def _norm_cell(v):
    """Make results comparable across trivially different types.

    SQLite may return 3 where gold returns 3.0, or Decimal-ish floats that
    differ in the last bit after a SUM. Neither is a semantic difference.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        return round(v, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def _norm_rows(rows) -> list[tuple]:
    return [tuple(_norm_cell(c) for c in row) for row in rows]


def _has_order_by(query: str) -> bool:
    return re.search(r"\border\s+by\b", query, re.IGNORECASE) is not None


def results_match(gold_rows, pred_rows, order_sensitive: bool) -> bool:
    g, p = _norm_rows(gold_rows), _norm_rows(pred_rows)
    if len(g) != len(p):
        return False
    if order_sensitive:
        return g == p
    # Order-insensitive over rows, but column order still matters: a query
    # that returns the right values under the wrong headers answered a
    # different question.
    return Counter(g) == Counter(p)


@dataclass
class EvalRow:
    db_id: str
    question: str
    gold: str
    pred: str
    correct: bool
    error: str | None = None
    error_kind: str | None = None  # for the error taxonomy table


def classify_error(err: Exception | None, correct: bool) -> str | None:
    if correct:
        return None
    if err is None:
        return "wrong_result"
    msg = str(err).lower()
    if isinstance(err, QueryTimeout):
        return "timeout"
    if "no such column" in msg:
        return "bad_column"
    if "no such table" in msg:
        return "bad_table"
    if "syntax error" in msg or "incomplete input" in msg:
        return "syntax"
    if "ambiguous" in msg:
        return "ambiguous_column"
    return "other_exec_error"


def score_one(
    db_root: str | Path,
    db_id: str,
    question: str,
    gold: str,
    raw_pred: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> EvalRow:
    pred = extract_sql(raw_pred)
    db_file = Path(db_root) / db_id / f"{db_id}.sqlite"

    if not pred:
        return EvalRow(db_id, question, gold, pred, False, "empty prediction", "empty")

    try:
        gold_rows = execute(db_file, gold, timeout_s)
    except Exception as e:  # noqa: BLE001
        # A gold query that will not run means the item cannot be scored;
        # surfacing it separately keeps it from silently counting as a miss.
        return EvalRow(db_id, question, gold, pred, False, f"gold failed: {e}", "gold_broken")

    try:
        pred_rows = execute(db_file, pred, timeout_s)
    except Exception as e:  # noqa: BLE001
        return EvalRow(db_id, question, gold, pred, False, str(e), classify_error(e, False))

    ok = results_match(gold_rows, pred_rows, _has_order_by(gold))
    return EvalRow(db_id, question, gold, pred, ok, None, classify_error(None, ok))


def aggregate(rows: list[EvalRow]) -> dict:
    scorable = [r for r in rows if r.error_kind != "gold_broken"]
    n = len(scorable) or 1
    correct = sum(r.correct for r in scorable)
    taxonomy = Counter(r.error_kind for r in scorable if r.error_kind)
    return {
        "n_total": len(rows),
        "n_scorable": len(scorable),
        "n_gold_broken": len(rows) - len(scorable),
        "n_correct": correct,
        "execution_accuracy": correct / n,
        "error_taxonomy": dict(taxonomy.most_common()),
    }
