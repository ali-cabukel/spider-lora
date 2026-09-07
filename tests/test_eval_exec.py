"""Offline tests for the eval harness. No model or GPU required.

    python -m pytest tests/ -q      (or just: python tests/test_eval_exec.py)
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_exec import (  # noqa: E402
    QueryTimeout,
    aggregate,
    execute,
    extract_sql,
    results_match,
    score_one,
)


def make_db(root: Path, db_id: str = "toy") -> Path:
    d = root / db_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{db_id}.sqlite"
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INT, country TEXT);
        CREATE TABLE song (id INTEGER PRIMARY KEY, singer_id INT, title TEXT, plays INT);
        INSERT INTO singer VALUES (1,'Ada',34,'UK'),(2,'Bo',51,'US'),(3,'Cy',29,'UK');
        INSERT INTO song VALUES (1,1,'Alpha',100),(2,1,'Beta',250),(3,2,'Gamma',70);
        """
    )
    con.commit()
    con.close()
    return p


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = make_db(root)

        # --- extract_sql -------------------------------------------------- #
        ok &= check(
            "strips markdown fences",
            extract_sql("```sql\nSELECT * FROM singer;\n```") == "SELECT * FROM singer",
        )
        ok &= check(
            "strips qwen think blocks",
            extract_sql("<think>hmm join?</think>SELECT name FROM singer")
            == "SELECT name FROM singer",
        )
        ok &= check(
            "strips a leading SQL: label",
            extract_sql("SQL: SELECT 1") == "SELECT 1",
        )
        ok &= check(
            "keeps only the first statement",
            extract_sql("SELECT 1; DROP TABLE singer;") == "SELECT 1",
        )
        ok &= check("collapses newlines", extract_sql("SELECT\n  name\nFROM singer") == "SELECT name FROM singer")

        # --- semantic equivalence ----------------------------------------- #
        g = execute(db, "SELECT name FROM singer WHERE country='UK'")
        p = execute(db, "SELECT name FROM singer WHERE 'UK'=country")
        ok &= check("predicate reordering counts as correct", results_match(g, p, False))

        g = execute(db, "SELECT name FROM singer ORDER BY age DESC")
        p = execute(db, "SELECT name FROM singer ORDER BY age ASC")
        ok &= check("ORDER BY direction is graded", not results_match(g, p, True))
        ok &= check("same rows unordered would pass without the flag", results_match(g, p, False))

        g = execute(db, "SELECT COUNT(*) FROM singer")
        p = execute(db, "SELECT 3.0")
        ok &= check("3 == 3.0 after normalization", results_match(g, p, False))

        g = execute(db, "SELECT name, age FROM singer WHERE id=1")
        p = execute(db, "SELECT age, name FROM singer WHERE id=1")
        ok &= check("column order still matters", not results_match(g, p, False))

        # --- scoring end to end ------------------------------------------- #
        r = score_one(root, "toy", "How many singers?", "SELECT COUNT(*) FROM singer", "```sql\nSELECT COUNT(*) FROM singer\n```")
        ok &= check("correct prediction scores 1", r.correct and r.error_kind is None)

        r = score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT COUNT(*) FROM singerz")
        ok &= check("hallucinated table is classified", not r.correct and r.error_kind == "bad_table")

        r = score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT nam FROM singer")
        ok &= check("hallucinated column is classified", r.error_kind == "bad_column")

        r = score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT FROM WHERE")
        ok &= check("syntax error is classified", r.error_kind == "syntax")

        r = score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "")
        ok &= check("empty prediction is classified", r.error_kind == "empty")

        r = score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT 999")
        ok &= check("runs but wrong answer is classified", r.error_kind == "wrong_result")

        r = score_one(root, "toy", "q", "SELECT * FROM nonexistent", "SELECT 1")
        ok &= check("unscorable gold is quarantined", r.error_kind == "gold_broken")

        # --- timeout ------------------------------------------------------- #
        bomb = (
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT COUNT(*) FROM c"
        )
        try:
            execute(db, bomb, timeout_s=1.0)
            ok &= check("runaway query is interrupted", False)
        except QueryTimeout:
            ok &= check("runaway query is interrupted", True)

        # --- aggregation --------------------------------------------------- #
        rows = [
            score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT COUNT(*) FROM singer"),
            score_one(root, "toy", "q", "SELECT COUNT(*) FROM singer", "SELECT 999"),
            score_one(root, "toy", "q", "SELECT * FROM nope", "SELECT 1"),
        ]
        agg = aggregate(rows)
        ok &= check(
            "gold_broken excluded from the denominator",
            agg["n_scorable"] == 2 and abs(agg["execution_accuracy"] - 0.5) < 1e-9,
        )

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
