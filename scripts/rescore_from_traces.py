"""Re-measure the deterministic checks of a past run — for free.

    python scripts/rescore_from_traces.py baseline-gen

Citation accuracy and abstention are computed from the answer text and the
retrieved context, both of which live in the trace. So when the *checker*
changes (a regex fix, a stricter abstention rule), the honest move is to
re-measure previous runs with the new checker rather than compare a new
measurement against an old one — and that costs zero LLM calls.

Prints the stored vs recomputed values and writes <tag>-rescored.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nom_rag.config import settings  # noqa: E402
from nom_rag.corpus.parse import load_chunks  # noqa: E402
from nom_rag.evalchecks import check_abstention, check_citations  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "evals" / "runs"


def load_traces() -> dict[str, dict]:
    traces: dict[str, dict] = {}
    if not settings.traces_dir.exists():
        return traces
    for path in sorted(settings.traces_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    traces[record["trace_id"]] = record
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Run tag to re-score, e.g. baseline-gen")
    args = parser.parse_args()

    run = json.loads((RUNS_DIR / f"{args.tag}.json").read_text(encoding="utf-8"))
    if not run.get("generation"):
        raise SystemExit(f"Run '{args.tag}' has no generation section to re-score.")

    traces = load_traces()
    chunks = {c.chunk_id: c for c in load_chunks()}

    rescored: dict[str, dict] = {}
    missing: list[str] = []
    for case_id, row in run["generation"]["per_case"].items():
        trace = traces.get(row.get("trace_id", ""))
        if trace is None:
            missing.append(case_id)
            continue
        retrieved = [chunks[r["chunk_id"]] for r in trace["retrieval"] if r["chunk_id"] in chunks]
        entry: dict = {"citations": None, "citation_accuracy": None, "abstained": None}
        if "abstained" in row:
            entry["abstained"] = check_abstention(trace["answer"])
        else:
            check = check_citations(trace["answer"], retrieved)
            entry |= {
                "citations": check.cited,
                "citation_accuracy": check.accuracy,
                "invalid_citations": check.invalid,
            }
        rescored[case_id] = entry

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    new_citation = mean(
        [e["citation_accuracy"] for e in rescored.values() if e["citation_accuracy"] is not None]
    )
    new_abstention = mean(
        [float(e["abstained"]) for e in rescored.values() if e["abstained"] is not None]
    )
    old = run["generation"]

    print(f"\n## re-score of '{args.tag}' (0 LLM calls)\n")
    print("| métrica | almacenado | recalculado |")
    print("|---|---|---|")
    print(f"| citation_accuracy | {old.get('citation_accuracy')} | {new_citation} |")
    print(f"| abstention_rate | {old.get('abstention_rate')} | {new_abstention} |")
    if missing:
        print(f"\nSin trace disponible ({len(missing)}): {', '.join(missing[:5])}")

    changed = [
        cid
        for cid, entry in rescored.items()
        if entry["citation_accuracy"] is not None
        and entry["citation_accuracy"] != old["per_case"].get(cid, {}).get("citation_accuracy")
    ]
    if changed:
        print(f"\nCasos con citation_accuracy distinto: {', '.join(changed)}")
        for cid in changed[:5]:
            before = old["per_case"][cid].get("citations")
            after = rescored[cid]["citations"]
            print(f"  {cid}: {before} -> {after}")

    out = RUNS_DIR / f"{args.tag}-rescored.json"
    out.write_text(
        json.dumps(
            {
                "tag": f"{args.tag}-rescored",
                "source_run": args.tag,
                "citation_accuracy": new_citation,
                "abstention_rate": new_abstention,
                "per_case": rescored,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
