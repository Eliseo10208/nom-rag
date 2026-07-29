"""Gemini implementation (google-genai SDK). Free-tier friendly: retries on
429/5xx with backoff honoring the server-suggested delay when present."""

from __future__ import annotations

import re
import time

from pydantic import BaseModel

from nom_rag.config import settings
from nom_rag.llm.base import LLMResponse

MAX_RETRIES = 4


class GeminiClient:
    def __init__(self, model: str, api_key: str | None = None):
        from google import genai

        key = api_key or settings.gemini_api_key
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and put it in .env"
            )
        self._client = genai.Client(api_key=key)
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max_tokens,
        )
        if response_model is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_model

        delay = 5.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except errors.APIError as exc:
                if exc.code not in (429, 500, 503) or attempt == MAX_RETRIES:
                    raise
                wait = _suggested_delay(str(exc)) or delay
                time.sleep(wait)
                delay = min(delay * 2, 60)

        text = resp.text or ""
        if not text:
            # Gemini 3.x models think by default and thinking tokens count
            # against max_output_tokens; when the budget runs out the response
            # carries no text. Fail loudly here instead of letting an empty
            # string surface as a mysterious schema-validation error upstream.
            finish = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
            raise RuntimeError(
                f"{self.model} returned no text (finish_reason={finish}). "
                "Raise max_tokens or lower the thinking budget."
            )

        usage = resp.usage_metadata
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )


def _suggested_delay(message: str) -> float | None:
    # 429 bodies include e.g. "retryDelay: 17s"
    m = re.search(r"retryDelay[\"':\s]+(\d+)", message)
    return float(m.group(1)) + 1 if m else None
