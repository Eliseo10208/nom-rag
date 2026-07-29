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
from nom_rag.index.store import RetrievedChunk, Retriever
from nom_rag.llm.base import LLMClient
from nom_rag.tracing import Trace, new_trace_id, write_trace

ABSTENTION_PHRASE = "No encuentro esa información en las normas indexadas"

SYSTEM_PROMPT = f"""Eres un asistente experto en Normas Oficiales Mexicanas (NOMs) de seguridad y \
salud en el trabajo emitidas por la STPS. Respondes preguntas usando EXCLUSIVAMENTE \
los extractos proporcionados como contexto.

Reglas obligatorias:
1. Cada afirmación debe llevar su cita con el numeral exacto entre corchetes, por \
ejemplo [NOM-035 §5.1] o [NOM-019 §7.4]. Solo puedes citar numerales que aparezcan \
en los extractos.
2. Si los extractos no contienen la información necesaria, responde exactamente: \
"{ABSTENTION_PHRASE}." y explica brevemente qué normas sí están indexadas. No inventes.
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


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n---\n\n".join(chunk.text for chunk in retrieved)
    return (
        f"Extractos de las NOMs (contexto):\n\n{context}\n\n"
        f"Pregunta del usuario: {question}"
    )


def ask(
    question: str,
    retriever: Retriever,
    client: LLMClient,
    k: int | None = None,
    write: bool = True,
) -> AskResult:
    t0 = time.perf_counter()
    retrieved = retriever.query(question, k=k)
    t1 = time.perf_counter()
    response = client.complete(build_prompt(question, retrieved), system=SYSTEM_PROMPT)
    t2 = time.perf_counter()

    citations = extract_citations(response.text)
    latency = {
        "retrieve": round((t1 - t0) * 1000),
        "generate": round((t2 - t1) * 1000),
        "total": round((t2 - t0) * 1000),
    }
    trace = Trace(
        trace_id=new_trace_id(),
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question=question,
        provider=settings.provider,
        model=response.model,
        retrieval=[{"chunk_id": c.chunk_id, "score": c.score} for c in retrieved],
        answer=response.text,
        citations=citations,
        usage={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        latency_ms=latency,
    )
    if write:
        write_trace(trace)
    return AskResult(
        answer=response.text,
        retrieved=retrieved,
        citations=citations,
        trace_id=trace.trace_id,
        model=response.model,
        usage=trace.usage,
        latency_ms=latency,
    )
