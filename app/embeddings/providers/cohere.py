from __future__ import annotations

from typing import Any, Dict, List

from app.embeddings.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.model_name = options.get("model", "embed-english-v3.0")
        self.api_key = options.get("api_key")
        self._client = None

    def _get_client(self):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is not installed. Run: pip install cohere")
        if self._client is None:
            self._client = cohere.Client(api_key=self.api_key)
        return self._client

    @property
    def dimension(self) -> int:
        if self.model_name in ("embed-english-v3.0", "embed-multilingual-v3.0"):
            return 1024
        elif self.model_name == "embed-english-light-v3.0":
            return 384
        return 1024

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        response = self._get_client().embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document",
        )
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self._get_client().embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query",
        )
        return response.embeddings[0]
