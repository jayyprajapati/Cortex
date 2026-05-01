from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.context import ExecutionContext
from app.embeddings.embedder import embed_query
from app.retrieval.bm25 import BM25
from app.vectorstore.qdrant_store import (
    list_user_document_ids,
    search_documents,
)

logger = logging.getLogger(__name__)


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _format_point(point) -> dict:
    payload = point.payload or {}
    return {
        "text": str(payload.get("text") or "").strip(),
        "page": payload.get("page"),
        "doc_id": payload.get("doc_id"),
        "section": payload.get("section"),
        "hierarchy": payload.get("hierarchy"),
        "token_count": payload.get("token_count"),
        "score": float(getattr(point, "score", 0.0)),
        "dense_score": float(getattr(point, "score", 0.0)),
    }


def retrieve(ctx: ExecutionContext, query: str) -> dict:
    """
    Hybrid retrieval: dense vector search + BM25 alpha-weighted re-scoring.

    Returns:
        {chunks, is_vague, doc_ids, clarification}
    """
    retrieval_cfg = ctx.registry.retrieval
    reranking_cfg = ctx.registry.reranking
    embed_cfg = ctx.registry.embedding
    collection = ctx.collection
    user_id = ctx.user_id
    requested_doc_ids = ctx.doc_ids

    t0 = time.monotonic()

    # --- Determine doc scope ---
    resolved_doc_ids: Optional[List[str]] = None
    if requested_doc_ids:
        resolved_doc_ids = [str(d).strip() for d in requested_doc_ids if str(d).strip()]
    else:
        known_docs = list_user_document_ids(collection, user_id)
        if not known_docs:
            return {
                "chunks": [],
                "is_vague": True,
                "doc_ids": [],
                "clarification": (
                    "No documents found in your account. "
                    "Please ingest documents first."
                ),
            }

    # --- Dense retrieval ---
    candidate_cap = (
        reranking_cfg.candidate_cap if reranking_cfg.enabled else retrieval_cfg.top_k * 3
    )
    candidate_cap = max(candidate_cap, retrieval_cfg.top_k * 2)

    query_vector = embed_query(query, embed_cfg)
    points = search_documents(
        query_vector=query_vector,
        collection_name=collection,
        user_id=user_id,
        doc_ids=resolved_doc_ids,
        top_k=candidate_cap,
    )

    chunks = [_format_point(p) for p in points]
    chunks = [c for c in chunks if c["text"]]

    if not chunks:
        return {
            "chunks": [],
            "is_vague": False,
            "doc_ids": resolved_doc_ids or [],
            "clarification": None,
        }

    # --- BM25 scoring on dense candidates ---
    if retrieval_cfg.hybrid and len(chunks) > 1:
        bm25 = BM25([c["text"] for c in chunks])
        bm25_raw = bm25.get_scores(query)

        dense_raw = [c["dense_score"] for c in chunks]
        norm_dense = _normalize(dense_raw)
        norm_bm25 = _normalize(bm25_raw)

        alpha = retrieval_cfg.alpha
        for i, chunk in enumerate(chunks):
            hybrid = alpha * norm_dense[i] + (1.0 - alpha) * norm_bm25[i]
            chunk["score"] = hybrid
            chunk["bm25_score"] = bm25_raw[i]

        chunks.sort(key=lambda c: c["score"], reverse=True)

    # Trim to requested top_k
    chunks = chunks[: retrieval_cfg.top_k]

    discovered_doc_ids = list({c["doc_id"] for c in chunks if c.get("doc_id")})
    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "retrieve app=%s user=%s hybrid=%s candidates=%d selected=%d latency_ms=%.1f",
        ctx.app_name,
        user_id,
        retrieval_cfg.hybrid,
        len(points),
        len(chunks),
        latency_ms,
    )

    return {
        "chunks": chunks,
        "is_vague": False,
        "doc_ids": discovered_doc_ids,
        "clarification": None,
    }
