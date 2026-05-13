from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class ChunkPayload(BaseModel):
    text: str
    doc_id: str
    chunk_id: int
    section_path: List[str]
    element_types: List[str] = []
    page: Optional[int] = None
    token_count: Optional[int] = None
    prev_chunk_id: Optional[int] = None
    next_chunk_id: Optional[int] = None
    user_id: str
    entity_hints: List[str] = []
    canonical_type: Optional[str] = None
    source_app: Optional[str] = None

    @field_validator("section_path")
    @classmethod
    def section_path_not_empty(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("section_path must not be empty; use ['_root'] for pre-heading content")
        return v


class BaseVectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, collection: str, dense_dim: int, sparse: bool = False) -> None:
        ...

    @abstractmethod
    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        payload: ChunkPayload,
    ) -> None:
        ...

    def upsert_batch(
        self,
        collection: str,
        points: List[Dict[str, Any]],
    ) -> None:
        """Batch upsert. Default falls back to individual upserts; providers should override."""
        for p in points:
            self.upsert(
                collection=collection,
                point_id=p["point_id"],
                vector=p["vector"],
                sparse_vector=p["sparse_vector"],
                payload=p["payload"],
            )

    @abstractmethod
    def search_dense(
        self,
        collection: str,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
        metadata_filter: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def search_sparse(
        self,
        collection: str,
        sparse_vector: Dict[str, Any],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def search_hybrid(
        self,
        collection: str,
        query_vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
        fusion: str,
        alpha: float,
        metadata_filter: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete_by_doc(self, collection: str, user_id: str, doc_id: str) -> int:
        ...

    @abstractmethod
    def delete_by_user(self, collection: str, user_id: str) -> int:
        ...

    @abstractmethod
    def list_docs(self, collection: str, user_id: str) -> List[str]:
        ...

    @abstractmethod
    def get_doc_chunks(
        self,
        collection: str,
        user_id: str,
        doc_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def drop_collection(self, collection: str) -> None:
        ...
