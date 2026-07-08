"""LLM-as-judge with structured verdicts.

The judge talks to the same LLMClient interface as the pipeline, so it can be
backed by Gemini, Claude, or a deterministic fake in harness tests. Verdicts
are Pydantic-validated JSON — a malformed judge response is an error, never a
silently-wrong metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from nom_rag.llm.base import LLMClient
from nom_rag.pipeline import AskResult

from .prompts import (
    CORRECTNESS_SYSTEM,
    CORRECTNESS_TEMPLATE,
    GROUNDEDNESS_SYSTEM,
    GROUNDEDNESS_TEMPLATE,
)


class GroundednessVerdict(BaseModel):
    reasoning: str = Field(description="Análisis afirmación por afirmación")
    unsupported_claims: list[str] = Field(
        description="Afirmaciones de la respuesta sin respaldo en los extractos"
    )
    grounded: bool = Field(description="true solo si unsupported_claims está vacía")


class CorrectnessVerdict(BaseModel):
    reasoning: str = Field(description="Comparación contra la referencia")
    correctness: int = Field(ge=1, le=5, description="1-5 según la rúbrica")


@dataclass
class JudgeVerdict:
    grounded: bool
    unsupported_claims: list[str]
    correctness: int
    groundedness_reasoning: str
    correctness_reasoning: str


class Judge:
    def __init__(self, client: LLMClient):
        self.client = client

    def evaluate(self, case, result: AskResult) -> JudgeVerdict:
        context = "\n\n---\n\n".join(chunk.text for chunk in result.retrieved)
        g_resp = self.client.complete(
            GROUNDEDNESS_TEMPLATE.format(
                context=context, question=case.question, answer=result.answer
            ),
            system=GROUNDEDNESS_SYSTEM,
            response_model=GroundednessVerdict,
            max_tokens=4096,
        )
        grounded = GroundednessVerdict.model_validate_json(g_resp.text)

        c_resp = self.client.complete(
            CORRECTNESS_TEMPLATE.format(
                question=case.question,
                reference=case.reference_answer,
                answer=result.answer,
            ),
            system=CORRECTNESS_SYSTEM,
            response_model=CorrectnessVerdict,
            max_tokens=4096,
        )
        correctness = CorrectnessVerdict.model_validate_json(c_resp.text)

        return JudgeVerdict(
            grounded=grounded.grounded,
            unsupported_claims=grounded.unsupported_claims,
            correctness=correctness.correctness,
            groundedness_reasoning=grounded.reasoning,
            correctness_reasoning=correctness.reasoning,
        )


def make_judge() -> Judge:
    from nom_rag.config import settings
    from nom_rag.llm.base import get_client

    return Judge(get_client(settings.provider, settings.judge_model))
