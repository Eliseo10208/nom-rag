"""Deterministic fake for tests and offline development. Returns canned
responses in FIFO order (or a fixed default) and records every call."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from nom_rag.llm.base import LLMResponse


@dataclass
class FakeLLM:
    responses: list[str] = field(default_factory=list)
    model: str = "fake-1"
    default: str = "Respuesta simulada. [NOM-035 §5.1]"
    calls: list[dict] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append(
            {"prompt": prompt, "system": system, "response_model": response_model}
        )
        text = self.responses.pop(0) if self.responses else self.default
        return LLMResponse(text=text, model=self.model, input_tokens=0, output_tokens=0)
