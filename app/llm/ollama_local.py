from __future__ import annotations

from typing import Optional

import requests

from app.llm.base import BaseLLM


class OllamaLocalLLM(BaseLLM):
    def __init__(self, model: str = "llama3") -> None:
        self._url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        response = requests.post(self._url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
