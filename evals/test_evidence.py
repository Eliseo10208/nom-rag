"""The retrieval score must not turn optional examples into obligations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nom_rag.evidence import EvidenceKind, evidence_kind, select_evidence
from nom_rag.llm.fake import FakeLLM
from nom_rag.pipeline import OUT_OF_CORPUS_ANSWER, ask

from .conftest import RETRIEVABLE_CASES


@dataclass
class _Chunk:
    numeral: str
    section_title: str
    text: str
    chunk_id: str = ""


def _real_chunks(*ids: str) -> list[_Chunk]:
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
    wanted = set(ids)
    found = []
    with corpus_path.open(encoding="utf-8") as corpus:
        for line in corpus:
            chunk = json.loads(line)
            if chunk["chunk_id"] in wanted:
                found.append(_Chunk(**{
                    key: chunk[key] for key in ("numeral", "section_title", "text", "chunk_id")
                }))
    assert {chunk.chunk_id for chunk in found} == wanted
    return found


def test_split_guide_and_non_normative_appendix_are_auxiliary():
    guide, appendix = _real_chunks("NOM-019:GR-I#2", "NOM-017:AP-I#3")
    assert evidence_kind(guide) is EvidenceKind.NON_NORMATIVE
    assert evidence_kind(appendix) is EvidenceKind.NON_NORMATIVE
    assert "no es de cumplimiento obligatorio" not in guide.text.lower()
    assert evidence_kind(_Chunk("AP-A", "Apéndice A. Normativo", "Contenido")) is EvidenceKind.NORMATIVE


def test_binding_question_excludes_optional_examples_and_flags_referral_only():
    chunks = _real_chunks(
        "NOM-019:GR-I#2", "NOM-019:GR-I#3", "NOM-017:AP-I#3",
        "NOM-017:5.3", "NOM-017:9.3#1",
    )
    decision = select_evidence("¿Qué requisitos establece la norma para trabajos en alturas?", chunks)

    assert {chunk.chunk_id for chunk in decision.excluded} == {
        "NOM-019:GR-I#2", "NOM-019:GR-I#3", "NOM-017:AP-I#3"
    }
    assert {chunk.chunk_id for chunk in decision.selected} == {"NOM-017:5.3", "NOM-017:9.3#1"}
    assert decision.should_abstain
    assert decision.instruction == ""


def test_current_top_five_for_out_of_scope_requirements_are_insufficient():
    # The actual top five from the checked-in index include both a guide's
    # hazard list and a binding clause that merely points to other NOMs.
    chunks = _real_chunks(
        "NOM-019:GR-I#2", "NOM-017:5.8", "NOM-017:5.3",
        "NOM-035:2", "NOM-030:GR-I",
    )
    decision = select_evidence("¿Qué requisitos establece la norma para trabajos en alturas?", chunks)

    assert {chunk.chunk_id for chunk in decision.excluded} == {
        "NOM-019:GR-I#2", "NOM-030:GR-I"
    }
    assert decision.should_abstain


def test_direct_normative_support_prevents_deterministic_abstention():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Protección contra caídas de altura: arnés")
    direct = _Chunk("5.1", "Obligaciones del patrón", "5.1 Debe proporcionar protección contra caídas de altura.")
    decision = select_evidence("¿Qué requisitos hay para protección contra caídas de altura?", [guide, direct])

    assert decision.selected == [direct]
    assert decision.excluded == [guide]
    assert not decision.should_abstain


def test_explicit_request_for_guide_keeps_it_with_nonbinding_instruction():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Ejemplos de peligros")
    decision = select_evidence("¿Qué ejemplos aparecen en la Guía de Referencia I?", [guide])

    assert decision.selected == [guide]
    assert not decision.excluded
    assert not decision.should_abstain
    assert "no conviertas sus listas en obligaciones" in decision.instruction


def test_what_guide_establishes_is_a_request_for_guide_content():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Trabajos en altura: ejemplos")
    decision = select_evidence("¿Qué establece la Guía de Referencia I sobre trabajos en altura?", [guide])

    assert decision.selected == [guide]
    assert "no conviertas sus listas en obligaciones" in decision.instruction


def test_guide_obligation_status_can_use_its_disclaimer():
    guide = _Chunk("GR-I", "Guía de Referencia I", "No es de cumplimiento obligatorio")
    decision = select_evidence("¿Es obligatoria la Guía de Referencia I?", [guide])

    assert decision.selected == [guide]
    assert decision.instruction


def test_requirement_request_takes_precedence_over_guide_mention():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Lista ilustrativa")
    decision = select_evidence("¿Qué requisitos de la guía debo cumplir para escaleras?", [guide])

    assert decision.selected == []
    assert decision.excluded == [guide]


def test_mandatory_measures_from_guide_are_not_treated_as_status_question():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Lista ilustrativa")
    decision = select_evidence("¿Qué medidas obligatorias tiene la guía para escaleras?", [guide])

    assert decision.selected == []
    assert decision.excluded == [guide]


def test_neutral_question_does_not_treat_unsolicited_guide_as_binding_evidence():
    guide = _Chunk("GR-I", "Guía de Referencia I", "Lista ilustrativa")
    decision = select_evidence("¿Qué pasa con las escaleras?", [guide])

    assert decision.selected == []
    assert decision.excluded == [guide]
    assert not decision.should_abstain


@pytest.mark.parametrize("question", [
    "¿Qué requisitos establece la norma para trabajos en alturas?",
    "¿Qué medidas de seguridad indica la NOM para trabajos en alturas?",
    "¿Qué medidas se aplican para trabajos en altura?",
])
def test_height_requirements_abstain_in_full_pipeline(retriever, question):
    """The original production question must never become a requirements list."""
    client = FakeLLM(default="Requisitos inventados [NOM-019 §4.2.1.6.1].")

    result = ask(
        question,
        retriever,
        client,
        write=False,
    )

    assert result.answer == OUT_OF_CORPUS_ANSWER
    assert result.citations == []
    assert result.model == "nom-rag-policy"
    assert client.calls == []


def test_retrievable_cases_keep_direct_normative_support(retrieval_results):
    false_abstentions = [
        case.id
        for case in RETRIEVABLE_CASES
        if select_evidence(case.question, retrieval_results.by_case[case.id]).should_abstain
    ]
    assert false_abstentions == []
