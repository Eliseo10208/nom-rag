"""Run the eval suite and persist a tagged, comparable snapshot.

    python scripts/run_evals.py --tag baseline
    python scripts/run_evals.py --tag candidate --generation

Retrieval metrics are deterministic and free. Generation metrics call the
configured LLM (generation + judge) and are opt-in via --generation.
Snapshots land in evals/runs/<tag>.json with the config that produced them;
compare two runs with scripts/compare_runs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from conftest import RETRIEVABLE_CASES, ALL_CASES  # noqa: E402

from nom_rag.config import settings  # noqa: E402
from nom_rag.refs import covered_refs  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "evals" / "runs"


def eval_retrieval(retriever, k: int) -> dict:
    per_case = {}
    for case in RETRIEVABLE_CASES:
        retrieved = retriever.query(case.question, k=k)
        covered = covered_refs(case.expected_refs, retrieved)
        first_rank = next(
            (
                rank
                for rank, chunk in enumerate(retrieved, start=1)
                if covered_refs(case.expected_refs, [chunk])
            ),
            None,
        )
        per_case[case.id] = {
            "hit": bool(covered),
            "recall": round(len(covered) / len(case.expected_refs), 3),
            "rr": round(1.0 / first_rank, 3) if first_rank else 0.0,
            "missing": sorted(set(case.expected_refs) - covered),
            "retrieved": [c.chunk_id for c in retrieved],
        }
    n = len(per_case)
    return {
        "k": k,
        "n_cases": n,
        "hit_rate": round(sum(m["hit"] for m in per_case.values()) / n, 3),
        "recall": round(sum(m["recall"] for m in per_case.values()) / n, 3),
        "mrr": round(sum(m["rr"] for m in per_case.values()) / n, 3),
        "per_case": per_case,
    }


def eval_generation(retriever, k: int) -> dict:
    from judges.judge import make_judge  # noqa: E402  (evals/ is on sys.path)
    from nom_rag.llm.base import get_client
    from nom_rag.pipeline import ask
    from nom_rag.evalchecks import check_abstention, check_citations

    client = get_client(settings.provider, settings.generation_model)
    judge = make_judge()
    per_case = {}
    for i, case in enumerate(ALL_CASES, 1):
        print(f"  [{i}/{len(ALL_CASES)}] {case.id}", file=sys.stderr)
        result = ask(case.question, retriever, client, k=k)
        row: dict = {
            "trace_id": result.trace_id,
            "citations": result.citations,
            "tokens": result.usage,
        }
        if case.retrievable:
            citation_check = check_citations(result.answer, result.retrieved)
            verdict = judge.evaluate(case, result)
            row |= {
                "citation_accuracy": citation_check.accuracy,
                "invalid_citations": citation_check.invalid,
                "grounded": verdict.grounded,
                "unsupported_claims": verdict.unsupported_claims,
                "correctness": verdict.correctness,
            }
        else:
            row["abstained"] = check_abstention(result.answer)
        per_case[case.id] = row

    scored = [r for r in per_case.values() if "correctness" in r]
    abstain = [r for r in per_case.values() if "abstained" in r]
    with_citations = [r for r in scored if r["citation_accuracy"] is not None]
    return {
        "n_scored": len(scored),
        "groundedness": round(sum(r["grounded"] for r in scored) / len(scored), 3),
        "correctness_avg": round(sum(r["correctness"] for r in scored) / len(scored), 3),
        "citation_accuracy": round(
            sum(r["citation_accuracy"] for r in with_citations) / len(with_citations), 3
        )
        if with_citations
        else None,
        "abstention_rate": round(sum(r["abstained"] for r in abstain) / len(abstain), 3)
        if abstain
        else None,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Run name, e.g. baseline")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument(
        "--generation", action="store_true", help="Also run generation + judge evals (calls the LLM)"
    )
    args = parser.parse_args()

    from nom_rag.corpus.parse import load_chunks
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import Retriever

    retriever = Retriever(Embedder(settings.embedding_model))
    snapshot = {
        "tag": args.tag,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "provider": settings.provider,
            "generation_model": settings.generation_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "k": args.k,
            "n_chunks": len(load_chunks()),
        },
        "retrieval": eval_retrieval(retriever, args.k),
        "generation": None,
    }
    if args.generation:
        snapshot["generation"] = eval_generation(retriever, args.k)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    r = snapshot["retrieval"]
    print(f"\n[{args.tag}] retrieval @k={r['k']}: "
          f"hit_rate={r['hit_rate']}  recall={r['recall']}  mrr={r['mrr']}")
    if snapshot["generation"]:
        g = snapshot["generation"]
        print(
            f"[{args.tag}] generation: groundedness={g['groundedness']}  "
            f"correctness={g['correctness_avg']}  citation_acc={g['citation_accuracy']}  "
            f"abstention={g['abstention_rate']}"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
