"""Offline regressions for the boundary of the intentionally small corpus."""

from __future__ import annotations

from dataclasses import dataclass, field

from nom_rag.llm.fake import FakeLLM
from nom_rag.pipeline import (
    INDEXED_NOM_IDS,
    OUT_OF_CORPUS_ANSWER,
    SYSTEM_PROMPT,
    ask,
    extract_nom_ids,
    unindexed_nom_ids,
)


@dataclass
class _RecordingRetriever:
    calls: list[tuple[str, int | None]] = field(default_factory=list)

    def query(self, query: str, k: int | None = None) -> list:
        self.calls.append((query, k))
        return []


def test_system_prompt_contains_the_exhaustive_corpus_catalog():
    for nom_id in INDEXED_NOM_IDS:
        assert f"- {nom_id}" in SYSTEM_PROMPT
    assert "referencias o la bibliografía" in SYSTEM_PROMPT
    assert OUT_OF_CORPUS_ANSWER in SYSTEM_PROMPT


def test_nom_identifier_extraction_normalizes_common_variants():
    assert extract_nom_ids("nom 007–ssa2–1993 y NOM-035") == [
        "NOM-007-SSA2-1993",
        "NOM-035",
    ]


def test_qualified_identifier_does_not_collide_with_same_nom_number():
    assert unindexed_nom_ids("¿Qué dice la NOM-017-SSA2-1994?") == [
        "NOM-017-SSA2-1994"
    ]
    assert unindexed_nom_ids("¿Qué dice la NOM-017-STPS-2024?") == []


def test_unindexed_nom_returns_canonical_answer_without_retrieval_or_generation():
    retriever = _RecordingRetriever()
    client = FakeLLM()

    result = ask(
        "¿Qué NOM es NOM-007-SSA2-1993 y qué dice?",
        retriever,
        client,
        write=False,
    )

    assert result.answer == OUT_OF_CORPUS_ANSWER
    assert result.retrieved == []
    assert result.citations == []
    assert result.model == "nom-rag-policy"
    assert result.usage == {"input_tokens": 0, "output_tokens": 0}
    assert retriever.calls == []
    assert client.calls == []


def test_indexed_nom_still_uses_retrieval_and_generation():
    retriever = _RecordingRetriever()
    client = FakeLLM(default="Está indexada.")

    result = ask("¿Qué dice la NOM-017?", retriever, client, write=False)

    assert result.answer == "Está indexada."
    assert retriever.calls == [("¿Qué dice la NOM-017?", None)]
    assert len(client.calls) == 1
    assert client.calls[0]["system"] == SYSTEM_PROMPT
