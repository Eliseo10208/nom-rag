"""Provider-agnostic LLM interface. The pipeline and the judge only ever see
this surface, so swapping Gemini for Claude (or a fake in tests) is a config
change, not a code change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Single completion. When response_model is given the provider must
        return JSON valid against that schema (the caller validates)."""
        ...


def get_client(provider: str, model: str) -> LLMClient:
    if provider == "gemini":
        from nom_rag.llm.gemini import GeminiClient

        return GeminiClient(model)
    if provider == "anthropic":
        from nom_rag.llm.anthropic_ import AnthropicClient

        return AnthropicClient(model)
    if provider == "fake":
        from nom_rag.llm.fake import FakeLLM

        return FakeLLM(model=model)
    raise ValueError(f"Unknown provider: {provider!r} (expected gemini | anthropic | fake)")
