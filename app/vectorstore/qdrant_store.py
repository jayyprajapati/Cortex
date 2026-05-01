from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import get_qdrant_client

logger = logging.getLogger(__name__)


def _client() -> QdrantClient:
    return get_qdrant_client()


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(collection_name: str, vector_size: int) -> None:
    """Idempotently ensure a collection exists with the correct vector size."""
    client = _client()

    try:
        info = client.get_collection(collection_name)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' already exists with vector size "
                f"{existing_size}, but the configured embedding model produces "
                f"{vector_size} dimensions. Delete the collection or use a matching model."
            )
        return
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" not in msg and "doesn't exist" not in msg and "does not exist" not in msg:
            raise

    _create_collection(client, collection_name, vector_size)


def create_collection(collection_name: str, vector_size: int) -> None:
    """Force-recreate a collection (drops existing data)."""
    client = _client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    _create_collection(client, collection_name, vector_size)


def _create_collection(client: QdrantClient, name: str, size: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Created collection '%s' (vector_size=%d)", name, size)


def list_collections() -> List[dict]:
    client = _client()
    result = []
    for col in client.get_collections().collections:
        try:
            info = client.get_collection(col.name)
            result.append({
                "name": col.name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "status": str(info.status),
            })
        except Exception:
            result.append({"name": col.name, "points_count": None, "vector_size": None, "status": "unknown"})
    return result


def get_collection_info(collection_name: str) -> dict:
    info = _client().get_collection(collection_name)
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "status": str(info.status),
    }


def get_collection_size(collection_name: str) -> int:
    return _client().get_collection(collection_name).points_count or 0


def delete_collection(collection_name: str) -> None:
    _client().delete_collection(collection_name)
    logger.info("Deleted collection '%s'", collection_name)


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def store_chunks(chunks: list, embeddings: list, user_id: str, collection_name: str) -> None:
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("user_id is required for chunk storage")

    points = []
    for chunk, vector in zip(chunks, embeddings):
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        if not doc_id:
            raise ValueError("doc_id is required for chunk storage")

        payload: dict = {
            "text": chunk.text,
            "doc_id": doc_id,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "user_id": normalized_user_id,
        }
        if getattr(chunk, "hierarchy", None):
            payload["hierarchy"] = chunk.hierarchy
        if getattr(chunk, "token_count", None) is not None:
            payload["token_count"] = chunk.token_count

        vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    _client().upsert(collection_name=collection_name, points=points)
    logger.info("Stored %d chunks in collection '%s'", len(points), collection_name)


def search_documents(
    query_vector: list,
    collection_name: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> list:
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("user_id is required for retrieval")

    must = [FieldCondition(key="user_id", match=MatchValue(value=normalized_user_id))]

    if doc_ids:
        clean = [str(d).strip() for d in doc_ids if str(d).strip()]
        if clean:
            if len(clean) == 1:
                must.append(FieldCondition(key="doc_id", match=MatchValue(value=clean[0])))
            else:
                must.append(FieldCondition(key="doc_id", match=MatchAny(any=clean)))

    response = _client().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=Filter(must=must),
        with_payload=True,
    )
    return response.points


def list_user_document_ids(
    collection_name: str, user_id: str, max_results: int = 1000
) -> List[str]:
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")

    query_filter = Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=normalized_user_id))]
    )
    client = _client()
    doc_ids: set = set()
    offset = None

    while len(doc_ids) < max_results:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            offset=offset,
            limit=100,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in points:
            doc_id = str((point.payload or {}).get("doc_id") or "").strip()
            if doc_id:
                doc_ids.add(doc_id)
        if next_offset is None:
            break
        offset = next_offset

    return sorted(doc_ids)


def scroll_document_chunks(
    collection_name: str,
    user_id: str,
    doc_id: Optional[str] = None,
    limit: int = 200,
) -> List[dict]:
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")

    must = [FieldCondition(key="user_id", match=MatchValue(value=normalized_user_id))]
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id).strip())))

    points, _ = _client().scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=must),
        limit=limit,
        with_payload=["text", "page", "doc_id", "section", "hierarchy", "token_count"],
        with_vectors=False,
    )

    chunks = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("text") or "").strip()
        if text:
            chunks.append({
                "text": text,
                "page": payload.get("page"),
                "doc_id": payload.get("doc_id"),
                "section": payload.get("section"),
                "hierarchy": payload.get("hierarchy"),
                "token_count": payload.get("token_count"),
                "score": 0.0,
            })
    return chunks


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def _build_delete_filter(user_id: str, doc_id: Optional[str] = None) -> Filter:
    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("user_id is required for deletion")

    must = [FieldCondition(key="user_id", match=MatchValue(value=normalized_user_id))]
    if doc_id is not None:
        clean_doc_id = str(doc_id).strip()
        if not clean_doc_id:
            raise ValueError("doc_id cannot be empty when provided for deletion")
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=clean_doc_id)))

    return Filter(must=must)


def _delete_by_filter(collection_name: str, delete_filter: Filter) -> int:
    client = _client()
    before_count = client.count(
        collection_name=collection_name,
        count_filter=delete_filter,
        exact=True,
    ).count

    if before_count == 0:
        raise ValueError("No matching vectors found to delete")

    client.delete(
        collection_name=collection_name,
        points_selector=delete_filter,
        wait=True,
    )

    after_count = client.count(
        collection_name=collection_name,
        count_filter=delete_filter,
        exact=True,
    ).count

    deleted = before_count - after_count
    if deleted <= 0:
        raise RuntimeError(
            f"Delete operation reported 0 removed vectors in collection '{collection_name}'"
        )
    return deleted


def delete_document_vectors(collection_name: str, user_id: str, doc_id: str) -> int:
    return _delete_by_filter(collection_name, _build_delete_filter(user_id=user_id, doc_id=doc_id))


def delete_user_vectors(collection_name: str, user_id: str) -> int:
    return _delete_by_filter(collection_name, _build_delete_filter(user_id=user_id))
