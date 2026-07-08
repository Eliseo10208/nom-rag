"""Compare two eval runs and print a markdown delta table.

    python scripts/compare_runs.py baseline candidate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "evals" / "runs"

RETRIEVAL_METRICS = ["hit_rate", "recall", "mrr"]
GENERATION_METRICS = ["groundedness", "correctness_avg", "citation_accuracy", "abstention_rate"]


def load(tag: str) -> dict:
    return json.loads((RUNS_DIR / f"{tag}.json").read_text(encoding="utf-8"))


def fmt_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "—"
    delta = b - a
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "＝")
    return f"{arrow} {delta:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("candidate")
    args = parser.parse_args()
    base, cand = load(args.base), load(args.candidate)

    print(f"\n## {args.base} → {args.candidate}\n")
    print("| métrica | " + args.base + " | " + args.candidate + " | Δ |")
    print("|---|---|---|---|")
    for metric in RETRIEVAL_METRICS:
        a, b = base["retrieval"].get(metric), cand["retrieval"].get(metric)
        print(f"| retrieval.{metric} | {a} | {b} | {fmt_delta(a, b)} |")
    if base.get("generation") and cand.get("generation"):
        for metric in GENERATION_METRICS:
            a, b = base["generation"].get(metric), cand["generation"].get(metric)
            print(f"| generation.{metric} | {a} | {b} | {fmt_delta(a, b)} |")

    base_cases = base["retrieval"]["per_case"]
    cand_cases = cand["retrieval"]["per_case"]
    fixed = [c for c in base_cases if not base_cases[c]["hit"] and cand_cases.get(c, {}).get("hit")]
    broke = [c for c in base_cases if base_cases[c]["hit"] and not cand_cases.get(c, {}).get("hit", True)]
    if fixed:
        print(f"\n**Casos que ahora aciertan:** {', '.join(fixed)}")
    if broke:
        print(f"\n**REGRESIONES (acertaban y ya no):** {', '.join(broke)}")
    if not fixed and not broke:
        print("\nSin cambios por caso en retrieval.")


if __name__ == "__main__":
    main()
