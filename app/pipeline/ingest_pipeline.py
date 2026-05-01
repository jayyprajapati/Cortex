from __future__ import annotations

import logging
import time
import uuid

from app.chunking.chunker import create_chunks
from app.context import ExecutionContext
from app.embeddings.embedder import embed_chunks, get_embedding_dimension
from app.ingestion.loader import load_document
from app.observability.logger import cortex_logger
from app.vectorstore.qdrant_store import ensure_collection, store_chunks

logger = logging.getLogger(__name__)


def resolve_doc_id(doc_id) -> str:
    normalized = str(doc_id).strip() if doc_id is not None else ""
    return normalized or str(uuid.uuid4())


def ingest_document(ctx: ExecutionContext, path: str, doc_id: str) -> dict:
    elements = load_document(path)
    return _ingest_elements(ctx, elements, doc_id)


def ingest_text(ctx: ExecutionContext, text: str, doc_id: str) -> dict:
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned:
        raise ValueError("text is required for raw text ingestion")
    return _ingest_elements(ctx, [{"text": cleaned, "page": 1}], doc_id)


def _ingest_elements(ctx: ExecutionContext, elements: list, doc_id: str) -> dict:
    ingestion_cfg = ctx.registry.ingestion
    embedding_cfg = ctx.registry.embedding
    collection = ctx.collection
    user_id = ctx.user_id
    t0 = time.monotonic()

    # --- Chunk ---
    t_chunk = time.monotonic()
    chunks = create_chunks(elements, doc_id, ingestion_cfg)
    chunk_ms = (time.monotonic() - t_chunk) * 1000

    for chunk in chunks:
        if hasattr(chunk, "source_app"):
            chunk.source_app = ctx.app_name

    if not chunks:
        raise ValueError(
            f"No chunks produced for doc_id={doc_id!r}. "
            "The document may be empty or unparseable with the configured strategy."
        )

    # --- Resolve embedding dimension and ensure collection ---
    dim = embedding_cfg.dimension or get_embedding_dimension(embedding_cfg.model)
    ensure_collection(collection, dim)

    # --- Embed ---
    t_embed = time.monotonic()
    embeddings = embed_chunks(chunks, embedding_cfg)
    embed_ms = (time.monotonic() - t_embed) * 1000

    # --- Store ---
    t_store = time.monotonic()
    store_chunks(chunks, embeddings, user_id, collection)
    store_ms = (time.monotonic() - t_store) * 1000

    total_ms = (time.monotonic() - t0) * 1000

    cortex_logger.log_ingest(
        app_name=ctx.app_name,
        doc_id=doc_id,
        user_id=user_id,
        strategy=ingestion_cfg.strategy,
        chunk_count=len(chunks),
        embed_model=embedding_cfg.model,
        collection=collection,
        chunk_latency_ms=chunk_ms,
        embed_latency_ms=embed_ms,
        store_latency_ms=store_ms,
        total_latency_ms=total_ms,
    )

    logger.info(
        "ingest complete doc_id=%s chunks=%d collection=%s total_ms=%.1f",
        doc_id,
        len(chunks),
        collection,
        total_ms,
    )

    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "collection": collection,
        "app_name": ctx.app_name,
    }
