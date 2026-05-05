from __future__ import annotations

from typing import Optional

from ollama import Client

from app.llm.base import BaseLLM


class OllamaCloudLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-oss:120b") -> None:
        self._client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
        )
        self.model = model

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature

        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options or None,
        )
        return response["message"]["content"]
