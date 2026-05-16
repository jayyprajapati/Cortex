from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI

from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Network/transient error types from the openai SDK that are safe to retry
_OPENAI_TRANSIENT_ERRORS = ("APIConnectionError", "APITimeoutError", "InternalServerError")

_REQUEST_TIMEOUT = 30  # seconds; override via subclass or env var in the future


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key, timeout=_REQUEST_TIMEOUT)
        self.model = model

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        last_exc: Optional[Exception] = None
        for attempt in range(2):  # 1 initial attempt + 1 retry
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                exc_type = type(exc).__name__
                is_transient = exc_type in _OPENAI_TRANSIENT_ERRORS
                last_exc = exc
                if is_transient and attempt == 0:
                    logger.warning(
                        "OpenAI transient error (attempt %d/2), retrying in 2s: %s",
                        attempt + 1, exc,
                    )
                    time.sleep(2)
                    continue
                raise

        raise last_exc  # type: ignore[misc]
