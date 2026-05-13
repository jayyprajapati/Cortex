from __future__ import annotations

from typing import Any, Dict, List

from app.embeddings.base import BaseEmbedder


class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.model_name = options.get("model", "BAAI/bge-small-en-v1.5")
        self.normalize = options.get("normalize", True)
        self.batch_size = options.get("batch_size", 32)
        self.passage_prefix = options.get("passage_prefix", "passage: ")
        self.query_prefix = options.get("query_prefix", "query: ")
        self._model = None

    def _get_model(self):
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        dim = self._get_model().get_sentence_embedding_dimension()
        if dim is None:
            return 384
        return dim

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        if self.passage_prefix:
            prefixed = [self.passage_prefix + t for t in texts]
        else:
            prefixed = texts
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        prefixed = self.query_prefix + text if self.query_prefix else text
        embedding = model.encode(
            [prefixed],
            normalize_embeddings=self.normalize,
            batch_size=1,
            show_progress_bar=False,
        )
        return embedding[0].tolist()
