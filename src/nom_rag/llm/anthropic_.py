"""Claude implementation. Ready to use once ANTHROPIC_API_KEY is set:

    NOMRAG_PROVIDER=anthropic
    NOMRAG_GENERATION_MODEL=claude-haiku-4-5
    NOMRAG_JUDGE_MODEL=claude-sonnet-5

Requires the optional dependency: pip install -e ".[anthropic]"
"""

from __future__ import annotations

from pydantic import BaseModel

from nom_rag.config import settings
from nom_rag.llm.base import LLMResponse


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install the extra first: pip install -e ".[anthropic]"') from exc

        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        if response_model is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": response_model.model_json_schema(),
                }
            }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
