"""Turn results/*_summary.json into the markdown tables for the README.

    python scripts/make_report.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_summaries() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*_summary.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"skipping malformed {p.name}")
    return out


def main() -> None:
    sums = load_summaries()
    if not sums:
        print("No results yet. Run src.evaluate first.")
        return

    print("### Execution accuracy (Spider dev)\n")
    print("| run | exec acc | correct / n | sec/example | backend |")
    print("|---|---|---|---|---|")
    for s in sorted(sums, key=lambda x: x.get("execution_accuracy", 0)):
        dev = s.get("backend", "").split("device=")[-1].split(" |")[0]
        print(
            f"| {s.get('tag','?')} | **{s.get('execution_accuracy',0):.3f}** | "
            f"{s.get('n_correct',0)} / {s.get('n_scorable',0)} | "
            f"{s.get('sec_per_example',0):.2f} | {dev} |"
        )

    print("\n### Error taxonomy\n")
    kinds = sorted({k for s in sums for k in s.get("error_taxonomy", {})})
    print("| run | " + " | ".join(kinds) + " |")
    print("|---" * (len(kinds) + 1) + "|")
    for s in sums:
        tax = s.get("error_taxonomy", {})
        print(f"| {s.get('tag','?')} | " + " | ".join(str(tax.get(k, 0)) for k in kinds) + " |")

    # Per-database breakdown from the best run: shows whether gains are broad
    # or concentrated in a few schemas.
    best = max(sums, key=lambda x: x.get("execution_accuracy", 0))
    pred_file = RESULTS / f"{best.get('tag')}_predictions.jsonl"
    if pred_file.exists():
        rows = [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]
        by_db: Counter = Counter()
        tot: Counter = Counter()
        for r in rows:
            tot[r["db_id"]] += 1
            by_db[r["db_id"]] += int(r["correct"])
        worst = sorted(by_db, key=lambda d: by_db[d] / tot[d])[:10]
        print(f"\n### Hardest databases ({best.get('tag')})\n")
        print("| db_id | acc | n |")
        print("|---|---|---|")
        for d in worst:
            print(f"| {d} | {by_db[d]/tot[d]:.2f} | {tot[d]} |")


if __name__ == "__main__":
    main()
