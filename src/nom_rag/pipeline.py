"""The RAG pipeline: retrieve -> prompt -> generate -> trace.

The prompt contract is load-bearing for the evals:
- every claim must cite an exact clause as [NOM-035 §5.1] (citation accuracy
  is checked deterministically against the retrieved context);
- out-of-corpus questions must be answered with the literal abstention phrase
  (abstention is also checked deterministically).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from nom_rag.config import settings
from nom_rag.corpus.sources import SOURCES
from nom_rag.evidence import select_evidence
from nom_rag.index.store import RetrievedChunk, Retriever
from nom_rag.llm.base import LLMClient
from nom_rag.refs import covered_refs
from nom_rag.tracing import Trace, new_trace_id, write_trace

ABSTENTION_PHRASE = "No encuentro esa información en las normas indexadas"

INDEXED_NOM_IDS = tuple(sorted(source.nom_id for source in SOURCES.values()))


def _join_spanish(items: tuple[str, ...]) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} y {items[-1]}"


INDEXED_NOM_CATALOG = _join_spanish(INDEXED_NOM_IDS)
OUT_OF_CORPUS_ANSWER = (
    f"{ABSTENTION_PHRASE}. Actualmente puedo consultar únicamente "
    f"{INDEXED_NOM_CATALOG}."
)

# Match both short identifiers (NOM-035) and qualified identifiers
# (NOM-007-SSA2-1993), accepting the dash variants people commonly paste.
_NOM_ID_RE = re.compile(
    r"\bNOM\s*[-–—]?\s*\d{3}"
    r"(?:\s*[-–—]\s*[A-Z][A-Z0-9]*)?"
    r"(?:\s*[-–—]\s*\d{4})?\b",
    re.IGNORECASE,
)


def extract_nom_ids(text: str) -> list[str]:
    """Return normalized, deduplicated NOM identifiers mentioned in text."""
    found: list[str] = []
    for match in _NOM_ID_RE.findall(text):
        normalized = re.sub(r"\s+", "", match.upper()).replace("–", "-").replace("—", "-")
        normalized = re.sub(r"^NOM(?=\d)", "NOM-", normalized)
        if normalized not in found:
            found.append(normalized)
    return found


def unindexed_nom_ids(text: str) -> list[str]:
    """Return explicit NOM identifiers that do not resolve to this corpus."""
    short_ids = set(SOURCES)
    full_ids = set(INDEXED_NOM_IDS)
    missing: list[str] = []
    for identifier in extract_nom_ids(text):
        if identifier in short_ids or identifier in full_ids:
            continue
        # Agency-qualified identifiers without a year still resolve when their
        # agency and number match a document in the corpus.
        if identifier.count("-") == 2 and any(
            full_id.startswith(f"{identifier}-") for full_id in full_ids
        ):
            continue
        missing.append(identifier)
    return missing

SYSTEM_PROMPT = f"""Eres un asistente experto en Normas Oficiales Mexicanas (NOMs) de seguridad y \
salud en el trabajo emitidas por la STPS. Respondes preguntas usando EXCLUSIVAMENTE \
los extractos proporcionados como contexto.

El corpus indexado contiene EXCLUSIVAMENTE estas normas:
{chr(10).join(f"- {nom_id}" for nom_id in INDEXED_NOM_IDS)}
Una NOM mencionada dentro de las referencias o la bibliografía de un extracto NO está
indexada a menos que aparezca en la lista anterior.

Reglas obligatorias:
1. Cada afirmación debe llevar su cita con el numeral exacto entre corchetes, por \
ejemplo [NOM-035 §5.1] o [NOM-019 §7.4]. Solo puedes citar numerales que aparezcan \
en los extractos.
2. Si los extractos no contienen la información necesaria, responde exactamente:
"{OUT_OF_CORPUS_ANSWER}"
No agregues otras normas ni intentes explicar una norma fuera del corpus.
3. Si la pregunta parte de una premisa incorrecta, corrígela explícitamente citando \
el numeral que la desmiente.
4. Responde en español, directo y conciso (máximo ~150 palabras), sin preámbulos.
"""

# A single bracket can contain several clause citations, sometimes repeating
# the NOM and sometimes inheriting it: "[NOM-035 §7.9; NOM-019 §9.6]" or
# "[NOM-035 §7.5, §7.6]". Parse every clause rather than only the first one.
_CITATION_LIKE_RE = re.compile(
    r"\[\s*(?:NOM\b|§\s*\d{1,2}(?:\.\d{1,3})+|\d{1,2}(?:\.\d{1,3})+)"
    r"[^\]]*\]",
    re.IGNORECASE,
)
_REF_TOKEN_RE = re.compile(
    r"(?:(?P<nom>NOM-\d{3})\s*§?\s*|(?P<inherited>§)\s*)"
    r"(?P<numeral>TRANS|GR-[IVXLC]+|AP-[A-Z0-9]+|\d{1,2}(?:\.\d{1,3})*)",
    re.IGNORECASE,
)
_NOM_MENTION_RE = re.compile(r"\bNOM\s*[-–—]?\s*\d{3}\b", re.IGNORECASE)
_UNPARSED_NUMERAL_RE = re.compile(r"[,;]\s*\d{1,2}(?:\.\d{1,3})*\b")

_SYMBOLIC_RE = re.compile(r"^(TRANS|GR-[IVXLC]+|AP-[A-Z0-9]+)", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3})*)")


def normalize_numeral(raw: str) -> str | None:
    """'7.5 a)' -> '7.5', '8.5.a' -> '8.5', '9.9b' -> '9.9', 'GR-I' -> 'GR-I'.
    Returns None when the text is not a clause reference at all."""
    ref = raw.strip().rstrip(".,;:").split("#")[0].strip()
    if match := _SYMBOLIC_RE.match(ref):
        return match.group(1).upper()
    if match := _NUMERIC_RE.match(ref):
        return match.group(1)
    return None


def _parse_citation_marker(marker: str) -> list[str] | None:
    """Parse every reference in one bracket, or reject an ambiguous marker."""
    body = marker[1:-1]
    tokens = list(_REF_TOKEN_RE.finditer(body))
    if not tokens:
        return None
    # A reference-shaped token that our grammar did not consume cannot be
    # silently ignored: it may be a second, unsupported citation.
    if len(_NOM_MENTION_RE.findall(body)) != sum(bool(t.group("nom")) for t in tokens):
        return None
    if body.count("§") != sum("§" in token.group() for token in tokens):
        return None
    remainder = body
    for token in reversed(tokens):
        remainder = remainder[: token.start()] + " " * len(token.group()) + remainder[token.end() :]
    if _UNPARSED_NUMERAL_RE.search(remainder):
        return None

    refs: list[str] = []
    current_nom: str | None = None
    for token in tokens:
        if token.group("nom"):
            current_nom = token.group("nom").upper()
        if current_nom is None:
            return None
        numeral = normalize_numeral(token.group("numeral"))
        if numeral is None:
            return None
        refs.append(f"{current_nom}:{numeral}")
    return refs


@dataclass
class AskResult:
    answer: str
    retrieved: list[RetrievedChunk]
    citations: list[str]  # ["NOM-035:5.1", ...]
    trace_id: str
    model: str
    usage: dict
    latency_ms: dict
    retrieval_query: str = ""  # differs from the question on rewritten follow-ups
    rewritten: bool = False


def extract_citations(answer: str) -> list[str]:
    """Clause references cited in an answer, in order, deduplicated. Several
    incisos of one clause collapse to a single citation — they are one claim
    about one clause, not five."""
    seen: list[str] = []
    for marker in _CITATION_LIKE_RE.finditer(answer):
        refs = _parse_citation_marker(marker.group())
        if refs is None:
            continue
        for ref in refs:
            if ref not in seen:
                seen.append(ref)
    return seen


def enforce_citation_contract(
    answer: str, retrieved: list[RetrievedChunk]
) -> tuple[str, list[str], str, list[str]]:
    """Return a safe answer, its supported refs, rejection reason and bad refs.

    A model can ignore the prompt and cite a clause it never saw, or answer
    without a parseable citation. In either case the exact abstention response
    is safer than showing an unsupported claim. The raw output is kept only in
    the trace for diagnosis.
    """
    if ABSTENTION_PHRASE.casefold() in answer.casefold():
        reason = "" if answer.strip() == OUT_OF_CORPUS_ANSWER else "normalized_abstention"
        return OUT_OF_CORPUS_ANSWER, [], reason, []

    for marker in _CITATION_LIKE_RE.finditer(answer):
        if _parse_citation_marker(marker.group()) is None:
            return OUT_OF_CORPUS_ANSWER, [], "unparseable_citation", []

    citations = extract_citations(answer)
    if not citations:
        return OUT_OF_CORPUS_ANSWER, [], "missing_citation", []

    covered = covered_refs(citations, retrieved)
    invalid = [ref for ref in citations if ref not in covered]
    if invalid:
        return OUT_OF_CORPUS_ANSWER, [], "unsupported_citation", invalid

    return answer, citations, "", []


def build_prompt(
    question: str, retrieved: list[RetrievedChunk], history: list | None = None
) -> str:
    context = "\n\n---\n\n".join(chunk.text for chunk in retrieved)
    # With no history the output is byte-identical to the single-turn prompt the
    # eval baseline was measured on — chat must not silently change that path.
    preface = ""
    if history:
        from nom_rag.conversation import format_history

        preface = (
            f"Conversación previa (solo como contexto; responde la última pregunta):\n"
            f"{format_history(history)}\n\n"
        )
    return (
        f"{preface}Extractos de las NOMs (contexto):\n\n{context}\n\n"
        f"Pregunta del usuario: {question}"
    )


def ask(
    question: str,
    retriever: Retriever,
    client: LLMClient,
    k: int | None = None,
    write: bool = True,
    history: list | None = None,
) -> AskResult:
    """Answer `question`. With `history`, the retrieval query is rewritten to be
    standalone first; the answer is still generated from the literal question."""
    t0 = time.perf_counter()
    query, rewritten = question, False
    guard_reason = ""
    rejected_answer = ""
    guard_invalid_refs: list[str] = []
    excluded_retrieval: list[dict] = []
    missing_ids = unindexed_nom_ids(question)
    if history and not missing_ids:
        from nom_rag.conversation import rewrite_question

        query, rewritten = rewrite_question(question, history, client)
        missing_ids = unindexed_nom_ids(query)

    if not missing_ids:
        from nom_rag.conversation import expand_retrieval_terms

        expanded_query = expand_retrieval_terms(query)
        rewritten = rewritten or expanded_query != query
        query = expanded_query

    if missing_ids:
        retrieved = []
        answer = OUT_OF_CORPUS_ANSWER
        model = "nom-rag-policy"
        provider = "local"
        usage = {"input_tokens": 0, "output_tokens": 0}
        citations: list[str] = []
        t1 = t2 = time.perf_counter()
    else:
        candidates = retriever.query(query, k=k)
        evidence = select_evidence(query, candidates)
        retrieved = evidence.selected
        excluded_retrieval = [
            {"chunk_id": chunk.chunk_id, "score": chunk.score}
            for chunk in evidence.excluded
        ]
        t1 = time.perf_counter()
        if evidence.should_abstain or not retrieved:
            answer = OUT_OF_CORPUS_ANSWER
            citations = []
            guard_reason = (
                "referral_only_evidence" if evidence.should_abstain else "no_normative_evidence"
            )
            model = "nom-rag-policy"
            provider = "local"
            usage = {"input_tokens": 0, "output_tokens": 0}
            t2 = t1
        else:
            system = SYSTEM_PROMPT
            if evidence.instruction:
                system += "\n" + evidence.instruction
            response = client.complete(
                build_prompt(question, retrieved, history), system=system
            )
            t2 = time.perf_counter()
            answer, citations, guard_reason, guard_invalid_refs = enforce_citation_contract(
                response.text, retrieved
            )
            if guard_reason:
                rejected_answer = response.text
            model = response.model
            provider = settings.provider
            usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }

    latency = {
        "retrieve": round((t1 - t0) * 1000),
        "generate": round((t2 - t1) * 1000),
        "total": round((t2 - t0) * 1000),
    }
    trace = Trace(
        trace_id=new_trace_id(),
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question=question,
        provider=provider,
        model=model,
        retrieval=[{"chunk_id": c.chunk_id, "score": c.score} for c in retrieved],
        answer=answer,
        citations=citations,
        usage=usage,
        latency_ms=latency,
        retrieval_query=query,
        rewritten=rewritten,
        guard_reason=guard_reason,
        guard_invalid_refs=guard_invalid_refs,
        rejected_answer=rejected_answer,
        excluded_retrieval=excluded_retrieval,
    )
    if write:
        write_trace(trace)
    return AskResult(
        answer=answer,
        retrieved=retrieved,
        citations=citations,
        trace_id=trace.trace_id,
        model=model,
        usage=usage,
        latency_ms=latency,
        retrieval_query=query,
        rewritten=rewritten,
    )
