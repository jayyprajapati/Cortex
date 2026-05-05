from __future__ import annotations

from typing import Optional

import requests

from app.config import OLLAMA_TIMEOUT
from app.llm.base import BaseLLM


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

        response = requests.post(self._url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]["content"]
