"""Follow-up handling: turn a context-dependent question into a standalone one.

Retrieval embeds the question, so "¿y con 30 empleados?" retrieves nothing —
its embedding carries none of the subject the user is actually asking about.
The fix is to rewrite it against the conversation before retrieving.

Two properties keep the cost and the blast radius small:
- the first turn has no history, so it is never rewritten (no LLM call);
- the rewrite only ever replaces the *retrieval query*. The answer is still
  generated from the user's literal question plus the retrieved context, so a
  bad rewrite degrades retrieval instead of silently changing what was asked.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pydantic import BaseModel, Field

from nom_rag.llm.base import LLMClient

MAX_HISTORY_TURNS = 4  # older turns rarely disambiguate and cost prompt tokens


@dataclass
class Turn:
    question: str
    answer: str


class RewrittenQuery(BaseModel):
    needs_rewrite: bool = Field(
        description="true solo si la pregunta depende del historial para entenderse"
    )
    standalone_question: str = Field(
        description="La pregunta reformulada de forma autónoma, o la original si no hacía falta"
    )


REWRITE_SYSTEM = """Reformulas preguntas de seguimiento para un buscador de normas oficiales \
mexicanas de seguridad laboral (NOM-035, NOM-019, NOM-017, NOM-030).

Recibes el historial de la conversación y la última pregunta del usuario. Tu tarea es \
devolver una pregunta AUTÓNOMA: que se entienda por sí sola, sin el historial.

Reglas:
- Sustituye pronombres y referencias implícitas ("eso", "ahí", "y si...", "¿y con...?") \
por el sujeto explícito que aparece en el historial.
- Conserva la intención y los datos concretos de la pregunta original (cifras, plazos).
- Si la pregunta ya se entiende sola, devuélvela sin cambios y marca needs_rewrite=false.
- Si se trata de recorridos de verificación fuera de programa, incluye también el
  término normativo "verificaciones extraordinarias" en la pregunta autónoma.
- NO respondas la pregunta. NO agregues información que el usuario no pidió.
- Responde en español."""


def expand_retrieval_terms(question: str) -> str:
    """Bridge colloquial off-schedule wording to the NOM's search vocabulary.

    This is deterministic and safe to use on a first turn without an LLM call.
    """
    folded = "".join(
        char
        for char in unicodedata.normalize("NFD", question.casefold())
        if not unicodedata.combining(char)
    )
    off_schedule = re.search(
        r"\b(?:fuera\s+de(?:l)?\s+programa(?:\s+anual)?|no\s+programad[aos]+)\b",
        folded,
    )
    verification = re.search(r"\b(?:recorrid\w*|verificac\w*)\b", folded)
    if off_schedule and verification and "extraordinari" not in folded:
        phrase = re.search(
            r"\b(?:fuera\s+de(?:l)?\s+programa(?:\s+anual)?|no\s+programad[ao]s?)\b",
            question,
            flags=re.IGNORECASE,
        )
        if phrase:
            before = question[: phrase.start()].rstrip()
            after = question[phrase.end() :]
            prefix = f"{before}, es decir " if before else ""
            return f"{prefix}verificaciones extraordinarias{after}"
    return question


def format_history(history: list[Turn], max_turns: int = MAX_HISTORY_TURNS) -> str:
    recent = history[-max_turns:]
    return "\n".join(
        f"Usuario: {turn.question}\nAsistente: {turn.answer}" for turn in recent
    )


def rewrite_question(
    question: str, history: list[Turn], client: LLMClient
) -> tuple[str, bool]:
    """Return (query_for_retrieval, was_rewritten). No history -> no LLM call."""
    if not history:
        return question, False

    prompt = (
        f"Historial de la conversación:\n{format_history(history)}\n\n"
        f"Última pregunta del usuario: {question}"
    )
    response = client.complete(
        prompt,
        system=REWRITE_SYSTEM,
        response_model=RewrittenQuery,
        max_tokens=1024,
    )
    result = RewrittenQuery.model_validate_json(response.text)
    rewritten = result.standalone_question.strip()
    query = rewritten if result.needs_rewrite and rewritten else question
    query = expand_retrieval_terms(query)
    return query, query != question
