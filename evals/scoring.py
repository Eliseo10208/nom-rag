"""Generation scoring shared by scripts/run_evals.py and test_generation.py —
one implementation so the CI gate and the tagged snapshots can never drift."""

from __future__ import annotations

import sys

from nom_rag.evalchecks import check_abstention, check_citations
from nom_rag.pipeline import ask


def score_generation(retriever, cases, k: int | None = None, progress: bool = True) -> dict:
    from nom_rag.config import settings
    from nom_rag.llm.base import get_client

    from .judges.judge import make_judge

    client = get_client(settings.provider, settings.generation_model)
    judge = make_judge()
    per_case: dict[str, dict] = {}
    for i, case in enumerate(cases, 1):
        if progress:
            print(f"  [{i}/{len(cases)}] {case.id}", file=sys.stderr)
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
        "groundedness": round(sum(r["grounded"] for r in scored) / len(scored), 3)
        if scored
        else None,
        "correctness_avg": round(sum(r["correctness"] for r in scored) / len(scored), 3)
        if scored
        else None,
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
