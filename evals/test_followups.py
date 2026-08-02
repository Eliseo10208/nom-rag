"""Evals for conversational follow-ups.

Adding query rewriting created a code path the original suite did not cover: a
question like "¿y con 30 empleados?" retrieves nothing on its own. These cases
measure whether rewriting recovers the right clauses, and the plumbing tests
below run offline so a broken rewriter fails CI without spending quota.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nom_rag.config import settings
from nom_rag.conversation import Turn, format_history, rewrite_question
from nom_rag.llm.fake import FakeLLM
from nom_rag.refs import covered_refs

DATASET = Path(__file__).parent / "dataset" / "followups.jsonl"


def load_followups() -> list[dict]:
    if not DATASET.exists():
        return []
    with DATASET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


FOLLOWUPS = load_followups()
# Measured 0.875 (7/8). The gate allows one more failure before tripping.
HIT_RATE_THRESHOLD = 0.75

# `s04` is retrieval-bound, not rewrite-bound: the user says "fuera de programa"
# while the norm says "verificaciones extraordinarias", and e5-small does not
# bridge that vocabulary gap. Diagnosed by checking that the golden phrasing of
# the same clause retrieves fine. Tracked, not hidden — same as the xfail cases
# in test_retrieval.py.
KNOWN_VOCABULARY_GAP = {"s04-extraordinarios"}


# ------------------------------------------------------- offline (no network)


def test_first_turn_never_calls_the_llm():
    """No history means nothing to disambiguate — spending a call would be waste."""
    fake = FakeLLM()
    query, rewritten = rewrite_question("¿Qué es la NOM-035?", [], fake)
    assert query == "¿Qué es la NOM-035?"
    assert rewritten is False
    assert fake.calls == [], "llamó al LLM sin historial"


def test_rewrite_uses_structured_output():
    fake = FakeLLM(
        responses=[
            json.dumps(
                {
                    "needs_rewrite": True,
                    "standalone_question": "¿Se debe evaluar el entorno organizacional con 30 empleados?",
                }
            )
        ]
    )
    history = [Turn("¿Cada cuánto se evalúa el entorno organizacional?", "Cada dos años.")]
    query, rewritten = rewrite_question("¿y con 30?", history, fake)
    assert rewritten is True
    assert "entorno organizacional" in query
    assert fake.calls[0]["response_model"] is not None, "el judge/rewriter debe usar schema"


def test_self_contained_question_is_left_alone():
    fake = FakeLLM(
        responses=[
            json.dumps(
                {"needs_rewrite": False, "standalone_question": "¿Qué dice la NOM-030?"}
            )
        ]
    )
    history = [Turn("Hola", "Hola")]
    query, rewritten = rewrite_question("¿Qué dice la NOM-030?", history, fake)
    assert query == "¿Qué dice la NOM-030?" and rewritten is False


def test_history_is_capped():
    turns = [Turn(f"p{i}", f"r{i}") for i in range(10)]
    text = format_history(turns, max_turns=3)
    assert "p9" in text and "p6" not in text


def test_single_turn_prompt_is_unchanged_by_the_chat_feature():
    """The eval baseline was measured without history; that prompt must stay
    byte-identical or the recorded metrics stop describing the shipped code."""
    from nom_rag.pipeline import build_prompt

    class _Chunk:
        text = "[NOM-035 §7.9] x"

    chunks = [_Chunk()]
    assert build_prompt("¿P?", chunks) == build_prompt("¿P?", chunks, history=None)
    assert build_prompt("¿P?", chunks) == build_prompt("¿P?", chunks, history=[])


def test_dataset_is_wellformed():
    assert FOLLOWUPS, "followups.jsonl vacío"
    ids = [c["id"] for c in FOLLOWUPS]
    assert len(ids) == len(set(ids))
    for case in FOLLOWUPS:
        assert case["history"], f"{case['id']}: sin historial, no es un seguimiento"
        assert case["expected_refs"], f"{case['id']}: sin cláusulas esperadas"


# ------------------------------------------------- live (needs the provider)


@pytest.mark.llm
@pytest.mark.skipif(
    settings.provider == "gemini" and not settings.gemini_api_key,
    reason="GEMINI_API_KEY not set",
)
def test_rewriting_recovers_the_right_clauses(retriever):
    """The metric that matters: after rewriting, does retrieval find the clause
    the follow-up was actually about?"""
    from nom_rag.llm.base import get_client

    client = get_client(settings.provider, settings.generation_model)
    hits, failures = 0, []
    for case in FOLLOWUPS:
        history = [Turn(t["question"], t["answer"]) for t in case["history"]]
        query, _ = rewrite_question(case["question"], history, client)
        retrieved = retriever.query(query)
        covered = covered_refs(case["expected_refs"], retrieved)
        missing_terms = [
            term for term in case.get("rewrite_must_mention", [])
            if term.lower() not in query.lower()
        ]
        if covered and not missing_terms:
            hits += 1
        else:
            failures.append(
                f"{case['id']}: query={query!r} covered={sorted(covered)} "
                f"missing_terms={missing_terms}"
            )

    hit_rate = hits / len(FOLLOWUPS)
    assert hit_rate >= HIT_RATE_THRESHOLD, (
        f"followup hit_rate={hit_rate:.2f}\n" + "\n".join(failures)
    )
