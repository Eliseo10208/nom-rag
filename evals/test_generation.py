"""Generation evals: LLM-as-judge + deterministic checks. Marked `llm` — they
call the configured provider (generation + judge) and need its API key.

    pytest evals -m llm

Gates were set from the first measured run (see evals/runs/) minus a margin;
groundedness and citation accuracy are the contract of this system — a drop
there means the pipeline started inventing content or citing thin air.
"""

from __future__ import annotations

import pytest

from nom_rag.config import settings

from .conftest import ALL_CASES

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        settings.provider == "gemini" and not settings.gemini_api_key,
        reason="GEMINI_API_KEY not set",
    ),
    pytest.mark.skipif(
        settings.provider == "fake", reason="generation evals need a real provider"
    ),
]

GROUNDEDNESS_THRESHOLD = 0.80
CORRECTNESS_THRESHOLD = 3.5
CITATION_ACCURACY_THRESHOLD = 0.90
ABSTENTION_THRESHOLD = 0.80


@pytest.fixture(scope="module")
def scores(retriever):
    from .scoring import score_generation

    return score_generation(retriever, ALL_CASES)


def test_groundedness(scores):
    failing = [
        cid
        for cid, row in scores["per_case"].items()
        if "grounded" in row and not row["grounded"]
    ]
    assert scores["groundedness"] >= GROUNDEDNESS_THRESHOLD, (
        f"groundedness={scores['groundedness']}; ungrounded cases={failing}"
    )


def test_correctness(scores):
    weak = [
        (cid, row["correctness"])
        for cid, row in scores["per_case"].items()
        if row.get("correctness", 5) <= 2
    ]
    assert scores["correctness_avg"] >= CORRECTNESS_THRESHOLD, (
        f"correctness_avg={scores['correctness_avg']}; worst={weak}"
    )


def test_citation_accuracy(scores):
    offenders = {
        cid: row["invalid_citations"]
        for cid, row in scores["per_case"].items()
        if row.get("invalid_citations")
    }
    assert scores["citation_accuracy"] >= CITATION_ACCURACY_THRESHOLD, (
        f"citation_accuracy={scores['citation_accuracy']}; invalid={offenders}"
    )


def test_abstention_on_out_of_scope(scores):
    failing = [
        cid
        for cid, row in scores["per_case"].items()
        if "abstained" in row and not row["abstained"]
    ]
    assert scores["abstention_rate"] >= ABSTENTION_THRESHOLD, (
        f"abstention_rate={scores['abstention_rate']}; answered anyway: {failing}"
    )
