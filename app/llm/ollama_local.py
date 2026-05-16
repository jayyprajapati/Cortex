from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from app.config import OLLAMA_TIMEOUT
from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLocalLLM(BaseLLM):
    def __init__(self, model: str = "llama3", base_url: str = "") -> None:
        url = (base_url or "http://localhost:11434").rstrip("/")
        self._url = f"{url}/api/chat"
        self.model = model

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "10m",
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        last_exc: Optional[Exception] = None
        for attempt in range(2):  # 1 initial attempt + 1 retry
            try:
                response = requests.post(self._url, json=payload, timeout=OLLAMA_TIMEOUT)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "OllamaLocal transient error (attempt %d/2), retrying in 2s: %s",
                        attempt + 1, exc,
                    )
                    time.sleep(2)
                    continue
                raise
            except Exception:
                raise

        raise last_exc  # type: ignore[misc]
