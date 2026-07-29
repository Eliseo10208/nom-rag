"""Generation scoring shared by scripts/run_evals.py and test_generation.py —
one implementation so the CI gate and the tagged snapshots can never drift.

Scoring is checkpointable: free-tier quotas are small enough that a full run
can hit a daily wall mid-suite, and losing an hour of completed cases to the
last API error is not acceptable. Each finished case is handed to `on_case`
so the caller can persist it, and `resume_from` skips cases already scored.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from nom_rag.evalchecks import check_abstention, check_citations
from nom_rag.pipeline import ask


class QuotaExhausted(RuntimeError):
    """Provider refused further calls (rate/quota). Partial results are valid."""


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def score_one(case, retriever, client, judge, k: int | None = None) -> dict:
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
    return row


def aggregate(per_case: dict[str, dict]) -> dict:
    scored = [r for r in per_case.values() if "correctness" in r]
    abstain = [r for r in per_case.values() if "abstained" in r]
    with_citations = [r for r in scored if r.get("citation_accuracy") is not None]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "n_scored": len(scored),
        "n_abstention_cases": len(abstain),
        "groundedness": mean([float(r["grounded"]) for r in scored]),
        "correctness_avg": mean([float(r["correctness"]) for r in scored]),
        "citation_accuracy": mean([r["citation_accuracy"] for r in with_citations]),
        "abstention_rate": mean([float(r["abstained"]) for r in abstain]),
        "per_case": per_case,
    }


def score_generation(
    retriever,
    cases,
    k: int | None = None,
    progress: bool = True,
    resume_from: dict[str, dict] | None = None,
    on_case: Callable[[str, dict, dict], None] | None = None,
) -> dict:
    """Score `cases`. Returns aggregate metrics plus per-case rows.

    `resume_from` seeds already-scored cases (skipped). `on_case(case_id, row,
    per_case)` fires after each newly scored case so callers can checkpoint.
    Raises QuotaExhausted (carrying partial results in .args[1]) when the
    provider cuts us off — the caller decides whether partial is good enough.
    """
    from nom_rag.config import settings
    from nom_rag.llm.base import get_client

    from .judges.judge import make_judge

    client = get_client(settings.provider, settings.generation_model)
    judge = make_judge()
    per_case: dict[str, dict] = dict(resume_from or {})

    pending = [c for c in cases if c.id not in per_case]
    if progress and per_case:
        print(f"  reanudando: {len(per_case)} casos ya medidos, faltan {len(pending)}",
              file=sys.stderr)

    for i, case in enumerate(pending, 1):
        if progress:
            print(f"  [{i}/{len(pending)}] {case.id}", file=sys.stderr)
        try:
            row = score_one(case, retriever, client, judge, k=k)
        except Exception as exc:
            if _is_quota_error(exc):
                raise QuotaExhausted(
                    f"quota agotada tras {len(per_case)} casos: {exc}", per_case
                ) from exc
            raise
        per_case[case.id] = row
        if on_case:
            on_case(case.id, row, per_case)

    return aggregate(per_case)
