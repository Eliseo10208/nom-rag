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
from nom_rag.index.store import RetrievedChunk, Retriever
from nom_rag.llm.base import LLMClient
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

# Capture the whole reference body and normalize it, rather than assuming a
# shape: models legitimately cite an inciso ("[NOM-035 §7.5 a)]", "§8.5.a"),
# and a checker that reads that as a clause named "7.5a" marks a CORRECT
# citation invalid. Incisos live inside their parent clause, which is the
# granularity the corpus is chunked and cited at, so they normalize to it.
CITATION_RE = re.compile(r"\[(NOM-\d{3})\s*§?\s*([^\]]{1,40})\]")

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
    for nom, raw in CITATION_RE.findall(answer):
        numeral = normalize_numeral(raw)
        if numeral is None:
            continue
        ref = f"{nom}:{numeral}"
        if ref not in seen:
            seen.append(ref)
    return seen


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
    missing_ids = unindexed_nom_ids(question)
    if history and not missing_ids:
        from nom_rag.conversation import rewrite_question

        query, rewritten = rewrite_question(question, history, client)
        missing_ids = unindexed_nom_ids(query)

    if missing_ids:
        retrieved = []
        answer = OUT_OF_CORPUS_ANSWER
        model = "nom-rag-policy"
        provider = "local"
        usage = {"input_tokens": 0, "output_tokens": 0}
        t1 = t2 = time.perf_counter()
    else:
        retrieved = retriever.query(query, k=k)
        t1 = time.perf_counter()
        response = client.complete(
            build_prompt(question, retrieved, history), system=SYSTEM_PROMPT
        )
        t2 = time.perf_counter()
        answer = response.text
        model = response.model
        provider = settings.provider
        usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

    citations = extract_citations(answer)
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
