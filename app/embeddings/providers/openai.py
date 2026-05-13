from __future__ import annotations

from typing import Any, Dict, List

from app.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.model_name = options.get("model", "text-embedding-3-small")
        self.api_key = options.get("api_key")
        self._client = None

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is not installed")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def dimension(self) -> int:
        if self.model_name == "text-embedding-3-large":
            return 3072
        return 1536

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        response = self._get_client().embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self._get_client().embeddings.create(model=self.model_name, input=[text])
        return response.data[0].embedding
