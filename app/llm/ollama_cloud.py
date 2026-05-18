from __future__ import annotations

import logging
import time
from typing import Optional

from ollama import Client

from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Transient error substrings that warrant a single retry
_TRANSIENT_MARKERS = ("connection", "timeout", "temporary", "503", "502", "500")


class OllamaCloudLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-oss:120b") -> None:
        self._client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
        )
        self.model = model

    def _is_transient(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(marker in msg for marker in _TRANSIENT_MARKERS)

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature

        last_exc: Optional[Exception] = None
        for attempt in range(2):  # 1 initial attempt + 1 retry
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options or None,
                )
                return response["message"]["content"]
            except Exception as exc:
                last_exc = exc
                if self._is_transient(exc) and attempt == 0:
                    logger.warning(
                        "OllamaCloud transient error (attempt %d/2), retrying in 2s: %s",
                        attempt + 1, exc,
                    )
                    time.sleep(2)
                    continue
                raise

        raise last_exc  # type: ignore[misc]
