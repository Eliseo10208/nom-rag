"""Deterministic retrieval evals: no LLM, no network, run on every push.

Aggregate gates protect against silent degradation when anything upstream
changes (chunker, embedding model, k, index build). The per-case test exists
for diagnosis: when an aggregate gate fails, the parametrized report shows
exactly which questions stopped retrieving.

Thresholds are set from the measured baseline (see evals/runs/) minus a small
margin — they are tripwires, not aspirations.
"""

from __future__ import annotations

import pytest

from nom_rag.refs import covered_refs

from .conftest import RETRIEVABLE_CASES

# measured @ e5-small, k=5 (run: split-definitions): hit=0.931 recall=0.842 mrr=0.722
# gates sit just under the measured values — any drop below them is a regression
HIT_RATE_THRESHOLD = 0.90
RECALL_THRESHOLD = 0.80
MRR_THRESHOLD = 0.68

def _case_metrics(case, retrieved) -> dict:
    covered = covered_refs(case.expected_refs, retrieved)
    first_rank = None
    for rank, chunk in enumerate(retrieved, start=1):
        if covered_refs(case.expected_refs, [chunk]):
            first_rank = rank
            break
    return {
        "hit": bool(covered),
        "recall": len(covered) / len(case.expected_refs),
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "missing": sorted(set(case.expected_refs) - covered),
    }


def test_hit_rate_at_k(retrieval_results):
    metrics = [
        _case_metrics(case, retrieval_results.by_case[case.id])
        for case in RETRIEVABLE_CASES
    ]
    hit_rate = sum(m["hit"] for m in metrics) / len(metrics)
    misses = [c.id for c, m in zip(RETRIEVABLE_CASES, metrics) if not m["hit"]]
    assert hit_rate >= HIT_RATE_THRESHOLD, f"hit_rate@k={hit_rate:.3f}; misses={misses}"


def test_recall_at_k(retrieval_results):
    recalls = [
        _case_metrics(case, retrieval_results.by_case[case.id])["recall"]
        for case in RETRIEVABLE_CASES
    ]
    recall = sum(recalls) / len(recalls)
    assert recall >= RECALL_THRESHOLD, f"recall@k={recall:.3f}"


def test_mrr(retrieval_results):
    rrs = [
        _case_metrics(case, retrieval_results.by_case[case.id])["rr"]
        for case in RETRIEVABLE_CASES
    ]
    mrr = sum(rrs) / len(rrs)
    assert mrr >= MRR_THRESHOLD, f"MRR={mrr:.3f}"


def test_document_retention_clause_is_retrieved_for_the_actual_question(retriever):
    question = "¿Cuánto tiempo debo conservar la documentación de la NOM-030?"
    retrieved = retriever.query(question)
    # §4.8 is a short obligation folded into the chunk headed §4.6.
    assert covered_refs(["NOM-030:4.8"], retrieved) == {"NOM-030:4.8"}


@pytest.mark.parametrize("case", RETRIEVABLE_CASES, ids=lambda c: c.id)
def test_case_diagnostics(case, retrieval_results):
    """Every known retrievable case must include a relevant clause in top-k."""
    metrics = _case_metrics(case, retrieval_results.by_case[case.id])
    if not metrics["hit"]:
        retrieved_ids = [c.chunk_id for c in retrieval_results.by_case[case.id]]
        message = f"{case.id}: none of {case.expected_refs} retrieved; got {retrieved_ids}"
        pytest.fail(message)
