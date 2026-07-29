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
sys.path.insert(0, str(PROJECT_ROOT))

from evals.conftest import ALL_CASES, RETRIEVABLE_CASES  # noqa: E402

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


def eval_generation(retriever, k: int, snapshot: dict, out_path: Path) -> dict:
    """Score generation, checkpointing to disk after every case so a quota wall
    (free tiers are small) costs at most one case of work, and a re-run with the
    same --tag resumes instead of starting over."""
    from evals.scoring import QuotaExhausted, aggregate, score_generation

    previous = (snapshot.get("generation") or {}).get("per_case") or {}

    def checkpoint(case_id: str, row: dict, per_case: dict) -> None:
        snapshot["generation"] = aggregate(per_case)
        snapshot["generation"]["complete"] = False
        out_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        result = score_generation(
            retriever, ALL_CASES, k=k, resume_from=previous, on_case=checkpoint
        )
        result["complete"] = len(result["per_case"]) == len(ALL_CASES)
        return result
    except QuotaExhausted as exc:
        partial = aggregate(exc.args[1])
        partial["complete"] = False
        print(
            f"\n!! Cuota agotada tras {len(partial['per_case'])}/{len(ALL_CASES)} casos.\n"
            f"   Progreso guardado. Reanuda con el MISMO --tag cuando haya cuota.",
            file=sys.stderr,
        )
        return partial


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

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.tag}.json"
    # a same-tag re-run resumes: keep whatever generation cases were already scored
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

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
        "generation": previous.get("generation"),
    }
    if args.generation:
        snapshot["generation"] = eval_generation(retriever, args.k, snapshot, out)

    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    r = snapshot["retrieval"]
    print(f"\n[{args.tag}] retrieval @k={r['k']}: "
          f"hit_rate={r['hit_rate']}  recall={r['recall']}  mrr={r['mrr']}")
    if snapshot["generation"]:
        g = snapshot["generation"]
        coverage = f"{len(g['per_case'])}/{len(ALL_CASES)} casos"
        status = "" if g.get("complete") else "  [PARCIAL — reanuda con el mismo --tag]"
        print(
            f"[{args.tag}] generation ({coverage}): groundedness={g['groundedness']}  "
            f"correctness={g['correctness_avg']}  citation_acc={g['citation_accuracy']}  "
            f"abstention={g['abstention_rate']}{status}"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
