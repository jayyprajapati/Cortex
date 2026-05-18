from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.vectorstore.base import BaseVectorStore, ChunkPayload


class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    def ensure_collection(self, collection: str, dense_dim: int, sparse: bool = False) -> None:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        payload: ChunkPayload,
    ) -> None:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def search_dense(
        self,
        collection: str,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
        metadata_filter: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def search_sparse(
        self,
        collection: str,
        sparse_vector: Dict[str, Any],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def search_hybrid(
        self,
        collection: str,
        query_vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
        fusion: str = "rrf",
        alpha: float = 0.5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def delete_by_doc(self, collection: str, user_id: str, doc_id: str) -> int:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def delete_by_user(self, collection: str, user_id: str) -> int:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def list_docs(self, collection: str, user_id: str) -> List[str]:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def get_doc_chunks(
        self,
        collection: str,
        user_id: str,
        doc_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Weaviate provider is not yet implemented")

    def drop_collection(self, collection: str) -> None:
        raise NotImplementedError("Weaviate provider is not yet implemented")
