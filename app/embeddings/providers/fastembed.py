from __future__ import annotations

from typing import Any, Dict, List

from app.embeddings.base import BaseEmbedder


class FastEmbedEmbedder(BaseEmbedder):
    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.model_name = options.get("model", "BAAI/bge-small-en-v1.5")
        self.sparse_model_name = options.get("sparse_model", "prithivida/Splade_PP_en_v1")
        self.normalize = options.get("normalize", True)
        self.batch_size = options.get("batch_size", 32)
        self._dense_model = None
        self._sparse_model = None

    def _get_dense_model(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed is not installed. Run: pip install fastembed")
        if self._dense_model is None:
            self._dense_model = TextEmbedding(model_name=self.model_name)
        return self._dense_model

    def _get_sparse_model(self):
        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise ImportError("fastembed is not installed. Run: pip install fastembed")
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    @property
    def dimension(self) -> int:
        if "bge-small" in self.model_name:
            return 384
        elif "bge-base" in self.model_name:
            return 768
        elif "openai" in self.model_name:
            return 1536
        return 384

    @property
    def supports_sparse(self) -> bool:
        return True

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._get_dense_model().embed(texts))
        return [v.tolist() for v in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return list(self._get_dense_model().query_embed([text]))[0].tolist()

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = list(self._get_sparse_model().embed(texts))
        return [{"indices": result.indices.tolist(), "values": result.values.tolist()} for result in results]
