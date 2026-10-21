"""Separate binding NOM clauses from illustrative material before generation.

The corpus gives reference guides and non-normative appendices ordinary chunk
IDs. A high similarity score does not make their examples legal requirements.
This module uses the parser's section metadata, not topic-specific exceptions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, Sequence, TypeVar


class EvidenceChunk(Protocol):
    numeral: str
    section_title: str
    text: str


T = TypeVar("T", bound=EvidenceChunk)


class EvidenceKind(str, Enum):
    NORMATIVE = "normative"
    NON_NORMATIVE = "non_normative"


NON_NORMATIVE_INSTRUCTION = (
    "Las Guías de Referencia y los apéndices identificados como no normativos "
    "son material orientativo. Si los citas, preséntalos expresamente como "
    "ejemplos o recomendaciones; no conviertas sus listas en obligaciones ni "
    "en requisitos de cumplimiento. Fundamenta obligaciones sólo en cláusulas "
    "normativas que establezcan directamente lo preguntado."
)


@dataclass
class EvidenceSelection(Generic[T]):
    selected: list[T]
    excluded: list[T]
    # True only when an obligation question's topic appears in retrieved
    # examples/guides or referrals to other applicable NOMs, without a direct
    # requirement in the remaining clauses. The caller should abstain.
    should_abstain: bool
    instruction: str


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def evidence_kind(chunk: EvidenceChunk) -> EvidenceKind:
    """Classify using section identity, which survives guide chunk splitting.

    AP-* is not automatically optional: a future normative appendix remains
    eligible unless its heading explicitly says it is non-normative.
    """
    numeral = chunk.numeral.upper()
    title = _fold(chunk.section_title)
    if numeral.startswith("GR-") or title.startswith("guia de referencia"):
        return EvidenceKind.NON_NORMATIVE
    if numeral.startswith("AP-") and "no normativo" in title:
        return EvidenceKind.NON_NORMATIVE
    return EvidenceKind.NORMATIVE


_HARD_BINDING_RE = re.compile(
    r"\b(?:requisitos?|requerimientos?|obligacion(?:es)?|obligatori[oa]s?|"
    r"cumplimiento|cumplir|exigencias?|exige(?:n)?|debe(?:n)?|"
    r"prescribe(?:n)?|mandat[oa]ri[oa]s?)\b"
)
_ESTABLISHES_RE = re.compile(r"\b(?:establece(?:n)?|establecid[oa]s?)\b")
_EXPLICIT_AUXILIARY_RE = re.compile(r"\b(?:guias?|apendices?)\b")
_ACTIONABLE_RE = re.compile(
    r"\b(?:medidas?|procedimientos?|precauciones?|lineamientos?|"
    r"protocolos?|acciones?|controles?|practicas?|protecciones?|"
    r"instrucciones?)\b"
)
_NORM_REFERENCE_RE = re.compile(r"\b(?:nom|normas?)\b")
_NORM_CONTENT_RE = re.compile(
    r"\b(?:indica(?:n)?|senala(?:n)?|dispone(?:n)?|aplica(?:n)?|"
    r"regula(?:n)?|contempla(?:n)?|dice|dicen)\b"
)
_ILLUSTRATIVE_RE = re.compile(
    r"\b(?:guias?|apendices?|ejemplos?|ilustrativ[oa]s?|"
    r"recomendaciones?|sugerencias?)\b"
)
_GUIDE_STATUS_RE = re.compile(r"\b(?:son|es|tienen|tiene|caracter)\b")
_GUIDE_STATUS_PROPERTY_RE = re.compile(r"\b(?:obligatori[oa]s?|normativ[oa]s?)\b")
_TOPIC_START_RE = re.compile(r"\b(?:para|sobre|respecto a|acerca de)\s+(.+)")
_TOPIC_STOPWORDS = frozenset(
    "de del el la los las un una unos unas en y o por segun que como "
    "trabajo trabajos actividad actividades centro centros empresa empresas "
    "norma normas nom requisitos requisito obligaciones obligacion debe deben "
    "establece establecen aplicables aplicable".split()
)


def _asks_for_binding_rule(question: str) -> bool:
    folded = _fold(question)
    explicit_auxiliary = bool(_EXPLICIT_AUXILIARY_RE.search(folded))
    # "¿Es obligatoria la guía?" asks about its status, not for obligations
    # supposedly imposed by it. The guide's disclaimer is relevant evidence.
    if (
        explicit_auxiliary
        and _GUIDE_STATUS_RE.search(folded)
        and _GUIDE_STATUS_PROPERTY_RE.search(folded)
        and not _ACTIONABLE_RE.search(folded)
        and not re.search(r"\b(?:requisitos?|obligacion(?:es)?|debe(?:n)?)\b", folded)
    ):
        return False
    if _HARD_BINDING_RE.search(folded):
        return True
    # "Qué establece la norma" asks for binding content, while "qué establece
    # la guía" asks for the guide's content and must retain its disclaimer.
    return not explicit_auxiliary and bool(
        _ESTABLISHES_RE.search(folded)
        or _ACTIONABLE_RE.search(folded)
        or (_NORM_REFERENCE_RE.search(folded) and _NORM_CONTENT_RE.search(folded))
    )


def _wants_illustrative_material(question: str) -> bool:
    folded = _fold(question)
    # A request for mandatory requirements takes precedence even if it mentions
    # the guide as a possible source (e.g. "requisitos de las guías").
    return bool(_ILLUSTRATIVE_RE.search(folded)) and not _asks_for_binding_rule(question)


def _topic_terms(question: str) -> tuple[str, ...]:
    """Extract a conservative explicit subject for the referral-only check.

    If the question has no clear `para/sobre/respecto a` subject, the check is
    inconclusive and leaves abstention to the existing answer contract.
    """
    folded = _fold(question)
    folded = re.sub(r"\bnom\s*[-–—]?\s*\d{3}\S*", "", folded)
    match = _TOPIC_START_RE.search(folded)
    if not match:
        return ()
    terms = [
        term for term in re.findall(r"[a-z0-9]+", match.group(1))
        if len(term) >= 4 and term not in _TOPIC_STOPWORDS
    ]
    return tuple(dict.fromkeys(terms[:5]))


def _line_mentions_topic(line: str, terms: tuple[str, ...]) -> bool:
    words = set(re.findall(r"[a-z0-9]+", _fold(line)))
    return bool(terms) and all(term in words for term in terms)


def _is_external_nom_example(line: str) -> bool:
    folded = _fold(line)
    return (
        "normas oficiales mexicanas aplicables" in folded
        and ("entre estas" in folded or "por ejemplo" in folded or "tales como" in folded)
    )


def _referral_only_subject(
    question: str, selected: Sequence[EvidenceChunk], excluded: Sequence[EvidenceChunk]
) -> bool:
    terms = _topic_terms(question)
    if not terms:
        return False
    indirect_mention = any(
        any(_line_mentions_topic(line, terms) for line in chunk.text.splitlines())
        for chunk in excluded
    )
    for chunk in selected:
        for line in chunk.text.splitlines():
            if not _line_mentions_topic(line, terms):
                continue
            if _is_external_nom_example(line):
                indirect_mention = True
            else:
                # A direct mention in a binding clause needs normal answer
                # scrutiny, so do not make a deterministic abstention decision.
                return False
    return indirect_mention


def select_evidence(question: str, chunks: Sequence[T]) -> EvidenceSelection[T]:
    """Filter retrieved context and identify a narrow referral-only abstention.

    Call with the standalone retrieval query after any conversation rewrite.
    Preserve `selected` as the LLM context; keep `excluded` for audit only.
    A caller can append `instruction` to its system prompt when non-normative
    material was explicitly requested.
    """
    if _wants_illustrative_material(question):
        selected = list(chunks)
        has_auxiliary = any(evidence_kind(chunk) is EvidenceKind.NON_NORMATIVE for chunk in selected)
        return EvidenceSelection(
            selected=selected,
            excluded=[],
            should_abstain=False,
            instruction=NON_NORMATIVE_INSTRUCTION if has_auxiliary else "",
        )

    selected = [chunk for chunk in chunks if evidence_kind(chunk) is EvidenceKind.NORMATIVE]
    excluded = [chunk for chunk in chunks if evidence_kind(chunk) is EvidenceKind.NON_NORMATIVE]
    should_abstain = _asks_for_binding_rule(question) and _referral_only_subject(
        question, selected, excluded
    )
    return EvidenceSelection(
        selected=selected,
        excluded=excluded,
        should_abstain=should_abstain,
        instruction="",
    )
