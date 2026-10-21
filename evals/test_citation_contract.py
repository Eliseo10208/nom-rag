"""Offline regressions for the runtime citation contract."""

from __future__ import annotations

from nom_rag.index.store import RetrievedChunk
from nom_rag.llm.fake import FakeLLM
from nom_rag.pipeline import (
    OUT_OF_CORPUS_ANSWER,
    ask,
    enforce_citation_contract,
    extract_citations,
)
from nom_rag.refs import chunk_covers
from nom_rag.tracing import Trace


def _chunk(numeral: str = "9.9", text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"NOM-019:{numeral}",
        nom_id="NOM-019-STPS-2011",
        short_id="NOM-019",
        numeral=numeral,
        section_title="Funcionamiento de las comisiones",
        text=text or f"[NOM-019 §{numeral}]\n{numeral} Verificaciones extraordinarias.",
        score=0.9,
    )


class _Retriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def query(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.chunks


def test_valid_citation_is_preserved():
    answer = "Las verificaciones extraordinarias se realizan en 30 días [NOM-019 §9.9]."
    assert enforce_citation_contract(answer, [_chunk()]) == (
        answer,
        ["NOM-019:9.9"],
        "",
        [],
    )


def test_missing_citation_becomes_exact_abstention_and_trace_keeps_original(monkeypatch):
    from nom_rag import pipeline

    written: list[Trace] = []
    monkeypatch.setattr(pipeline, "write_trace", written.append)
    raw = "Deben realizarse dentro de treinta días."
    result = ask("¿Cuándo se hacen?", _Retriever([_chunk()]), FakeLLM(default=raw))

    assert result.answer == OUT_OF_CORPUS_ANSWER
    assert result.citations == []
    assert written[0].answer == OUT_OF_CORPUS_ANSWER
    assert written[0].guard_reason == "missing_citation"
    assert written[0].rejected_answer == raw
    assert written[0].retrieval == [{"chunk_id": "NOM-019:9.9", "score": 0.9}]


def test_unretrieved_nom_019_9_6_is_rejected_even_with_another_valid_citation():
    raw = "Se hacen fuera de programa [NOM-019 §9.6] y en 30 días [NOM-019 §9.9]."
    answer, citations, reason, invalid = enforce_citation_contract(raw, [_chunk()])

    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "unsupported_citation"
    assert invalid == ["NOM-019:9.6"]


def test_every_reference_inside_one_bracket_is_checked():
    raw = "Cada dos años [NOM-035 §7.9; NOM-019 §9.6]."
    valid = RetrievedChunk(
        chunk_id="NOM-035:7.9",
        nom_id="NOM-035-STPS-2018",
        short_id="NOM-035",
        numeral="7.9",
        section_title="Evaluación",
        text="7.9 Cada dos años.",
        score=0.9,
    )

    assert extract_citations(raw) == ["NOM-035:7.9", "NOM-019:9.6"]
    answer, citations, reason, invalid = enforce_citation_contract(raw, [valid])
    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "unsupported_citation"
    assert invalid == ["NOM-019:9.6"]


def test_multiple_clauses_of_one_nom_and_inciso_remain_valid():
    raw = "Deben cumplirse [NOM-019 §9.5, §9.9]."
    assert extract_citations(raw) == ["NOM-019:9.5", "NOM-019:9.9"]
    assert extract_citations("Según [NOM-035 §7.5 a)] y [NOM-035 §8.5.a].") == [
        "NOM-035:7.5",
        "NOM-035:8.5",
    ]
    chunks = [_chunk("9.5"), _chunk("9.9")]
    assert enforce_citation_contract(raw, chunks) == (
        raw,
        ["NOM-019:9.5", "NOM-019:9.9"],
        "",
        [],
    )


def test_ambiguous_extra_numeral_in_one_bracket_is_rejected():
    raw = "Cada dos años [NOM-019 §9.9, 9.6]."
    answer, citations, reason, _ = enforce_citation_contract(raw, [_chunk("9.9")])
    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "unparseable_citation"


def test_malformed_bare_numeral_is_rejected_even_with_valid_citation():
    raw = "Punto de anclaje [4.2.1.6.1]; verificaciones [NOM-019 §9.9]."
    answer, citations, reason, _ = enforce_citation_contract(raw, [_chunk()])

    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "unparseable_citation"


def test_non_normative_guide_items_cannot_cover_numeric_clause_citations():
    guide = _chunk(
        "GR-I#2",
        "[NOM-019 §GR-I] Guía de Referencia I\n4.2.1.6.1 Punto de anclaje.",
    )
    assert chunk_covers(guide.short_id, guide.numeral, guide.text, "NOM-019:GR-I")
    assert not chunk_covers(
        guide.short_id, guide.numeral, guide.text, "NOM-019:4.2.1.6.1"
    )
    answer, citations, reason, invalid = enforce_citation_contract(
        "Punto de anclaje [NOM-019 §4.2.1.6.1].", [guide]
    )
    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "unsupported_citation"
    assert invalid == ["NOM-019:4.2.1.6.1"]


def test_abstention_is_canonical_and_uncited():
    assert enforce_citation_contract(OUT_OF_CORPUS_ANSWER, [_chunk()]) == (
        OUT_OF_CORPUS_ANSWER,
        [],
        "",
        [],
    )
    answer, citations, reason, _ = enforce_citation_contract(
        OUT_OF_CORPUS_ANSWER + " Pero tal vez sea mensual [NOM-019 §9.6].",
        [_chunk()],
    )
    assert answer == OUT_OF_CORPUS_ANSWER
    assert citations == []
    assert reason == "normalized_abstention"


def test_old_trace_records_remain_readable():
    trace = Trace(
        trace_id="old",
        ts="2026-01-01T00:00:00+00:00",
        question="¿Pregunta?",
        provider="gemini",
        model="m",
        retrieval=[],
        answer=OUT_OF_CORPUS_ANSWER,
        citations=[],
    )
    assert trace.guard_reason == ""
    assert trace.guard_invalid_refs == []
    assert trace.rejected_answer == ""
