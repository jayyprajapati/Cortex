from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.vectorstore.base import BaseVectorStore, ChunkPayload


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self._client = None

    def _get_client(self):
        from app.config import get_qdrant_client
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def _build_filter(
        self,
        user_id: str,
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None,
    ):
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]

        if doc_ids:
            clean = [str(d).strip() for d in doc_ids if str(d).strip()]
            if clean:
                if len(clean) == 1:
                    must.append(FieldCondition(key="doc_id", match=MatchValue(value=clean[0])))
                else:
                    must.append(FieldCondition(key="doc_id", match=MatchAny(any=clean)))

        if metadata_filter:
            for k, v in metadata_filter.items():
                must.append(FieldCondition(key=k, match=MatchValue(value=v)))

        return Filter(must=must)

    def _format_point(self, point) -> Dict:
        payload = point.payload or {}
        return {
            "text": str(payload.get("text") or "").strip(),
            "page": payload.get("page"),
            "doc_id": payload.get("doc_id"),
            "chunk_id": payload.get("chunk_id"),
            "section_path": payload.get("section_path", ["_root"]),
            "section": (payload.get("section_path") or ["_root"])[-1],
            "element_types": payload.get("element_types", []),
            "token_count": payload.get("token_count"),
            "prev_chunk_id": payload.get("prev_chunk_id"),
            "next_chunk_id": payload.get("next_chunk_id"),
            "entity_hints": payload.get("entity_hints", []),
            "canonical_type": payload.get("canonical_type"),
            "score": float(getattr(point, "score", 0.0)),
        }

    def ensure_collection(self, collection: str, dense_dim: int, sparse: bool = False) -> None:
        from qdrant_client.models import (
            Distance,
            VectorParams,
            SparseVectorParams,
            PayloadSchemaType,
        )

        client = self._get_client()

        try:
            info = client.get_collection(collection)
            # Collection exists — check dense dimension
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                # Named vectors config
                if "dense" in vectors_config:
                    existing_dim = vectors_config["dense"].size
                    if existing_dim != dense_dim:
                        raise ValueError(
                            f"Collection '{collection}' dense dimension mismatch: "
                            f"existing={existing_dim}, requested={dense_dim}"
                        )
            else:
                # Plain vectors config
                existing_dim = vectors_config.size
                if existing_dim != dense_dim:
                    raise ValueError(
                        f"Collection '{collection}' dense dimension mismatch: "
                        f"existing={existing_dim}, requested={dense_dim}"
                    )
            return
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" not in msg and "doesn't exist" not in msg and "status_code=404" not in msg:
                raise

        # Collection does not exist — create it
        if sparse:
            client.create_collection(
                collection_name=collection,
                vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
        else:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dense_dim, distance=Distance.COSINE),
            )

        # Create payload indexes
        for field in ("user_id", "doc_id", "canonical_type"):
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        payload: ChunkPayload,
    ) -> None:
        self.upsert_batch(collection, [{"point_id": point_id, "vector": vector, "sparse_vector": sparse_vector, "payload": payload}])

    def upsert_batch(
        self,
        collection: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        client = self._get_client()

        qdrant_points = []
        for p in points:
            vector = p["vector"]
            sparse_vector = p["sparse_vector"]
            payload_dict = p["payload"].model_dump()

            if sparse_vector is not None:
                vectors = {
                    "dense": vector,
                    "sparse": SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                }
                point = PointStruct(id=p["point_id"], vector=vectors, payload=payload_dict)
            else:
                point = PointStruct(id=p["point_id"], vector=vector, payload=payload_dict)
            qdrant_points.append(point)

        for i in range(0, len(qdrant_points), batch_size):
            client.upsert(collection_name=collection, points=qdrant_points[i:i + batch_size])

    def search_dense(
        self,
        collection: str,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
        metadata_filter: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        client = self._get_client()
        filter_obj = self._build_filter(user_id, doc_ids, metadata_filter)

        try:
            response = client.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense",
                limit=top_k,
                query_filter=filter_obj,
                with_payload=True,
            )
        except Exception:
            response = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                query_filter=filter_obj,
                with_payload=True,
            )

        return [self._format_point(p) for p in response.points]

    def search_sparse(
        self,
        collection: str,
        sparse_vector: Dict[str, Any],
        user_id: str,
        top_k: int,
        doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import SparseVector

        client = self._get_client()
        filter_obj = self._build_filter(user_id, doc_ids)

        response = client.query_points(
            collection_name=collection,
            query=SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="sparse",
            limit=top_k,
            query_filter=filter_obj,
            with_payload=True,
        )

        return [self._format_point(p) for p in response.points]

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
        if sparse_vector is None:
            return self.search_dense(collection, query_vector, user_id, top_k, doc_ids, metadata_filter)

        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

        client = self._get_client()
        filter_obj = self._build_filter(user_id, doc_ids, metadata_filter)

        prefetches = [
            Prefetch(query=query_vector, using="dense", limit=top_k * 3),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using="sparse",
                limit=top_k * 3,
            ),
        ]

        fusion_type = Fusion.RRF if fusion == "rrf" else Fusion.DBSF

        response = client.query_points(
            collection_name=collection,
            prefetch=prefetches,
            query=FusionQuery(fusion=fusion_type),
            limit=top_k,
            query_filter=filter_obj,
            with_payload=True,
        )

        return [self._format_point(p) for p in response.points]

    def delete_by_doc(self, collection: str, user_id: str, doc_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        filter_obj = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
        )

        try:
            count_result = client.count(collection_name=collection, count_filter=filter_obj)
            deleted_count = count_result.count
            if deleted_count > 0:
                client.delete(collection_name=collection, points_selector=filter_obj)
            return deleted_count
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "status_code=404" in msg:
                return 0
            raise

    def delete_by_user(self, collection: str, user_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        filter_obj = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        try:
            count_result = client.count(collection_name=collection, count_filter=filter_obj)
            deleted_count = count_result.count
            if deleted_count > 0:
                client.delete(collection_name=collection, points_selector=filter_obj)
            return deleted_count
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "status_code=404" in msg:
                return 0
            raise

    def list_docs(self, collection: str, user_id: str) -> List[str]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        filter_obj = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        doc_ids: set[str] = set()
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=filter_obj,
                limit=100,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )

            for record in records:
                if record.payload and record.payload.get("doc_id"):
                    doc_ids.add(str(record.payload["doc_id"]))

            if next_offset is None:
                break
            offset = next_offset

        return list(doc_ids)

    def get_doc_chunks(
        self,
        collection: str,
        user_id: str,
        doc_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if doc_id:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

        filter_obj = Filter(must=must)

        records, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filter_obj,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [self._format_point(r) for r in records]

    def drop_collection(self, collection: str) -> None:
        client = self._get_client()
        client.delete_collection(collection)
