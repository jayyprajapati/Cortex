from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

from app.context import ExecutionContext
from app.observability.logger import cortex_logger
from app.vectorstore.base import ChunkPayload

logger = logging.getLogger(__name__)


def resolve_doc_id(doc_id) -> str:
    normalized = str(doc_id).strip() if doc_id is not None else ""
    return normalized or str(uuid.uuid4())


def ingest_document(ctx: ExecutionContext, path: str, doc_id: str) -> dict:
    """Ingest a file at the given path using ctx.components.loader."""
    loader = ctx.components.loader
    elements = loader.load(path)
    return _ingest_elements(ctx, elements, doc_id)


def ingest_text(ctx: ExecutionContext, text: str, doc_id: str) -> dict:
    """Ingest raw text as a single document."""
    from app.ingestion.loaders.base import Element
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned:
        raise ValueError("text is required for raw text ingestion")
    # Treat raw text as a single paragraph element
    elements = [Element(type="paragraph", text=cleaned, page=1)]
    return _ingest_elements(ctx, elements, doc_id)


def _ingest_elements(ctx: ExecutionContext, elements, doc_id: str) -> dict:
    """Core ingestion: chunk → embed → store via ctx.components."""
    chunking_cfg = ctx.registry.chunking
    embedding_cfg = ctx.registry.embedding
    collection = ctx.collection
    user_id = ctx.user_id
    t0 = time.monotonic()

    # --- Chunk ---
    t_chunk = time.monotonic()
    chunks = ctx.components.chunker.chunk(elements, doc_id)
    chunk_ms = (time.monotonic() - t_chunk) * 1000

    if not chunks:
        raise ValueError(
            f"No chunks produced for doc_id={doc_id!r}. "
            "The document may be empty or unparseable with the configured strategy."
        )

    # --- Annotate chunks with source_app ---
    for chunk in chunks:
        if hasattr(chunk, "source_app"):
            chunk.source_app = ctx.app_name

    # --- Extract entity hints (cheap regex pass) ---
    try:
        from app.chunking.entity_hints import extract_entity_hints
        # entity_hints expects List[dict] with a 'text' key; convert elements if needed
        elements_for_hints = []
        for el in list(elements)[:5]:
            if isinstance(el, dict):
                elements_for_hints.append(el)
            else:
                elements_for_hints.append({"text": getattr(el, "text", ""), "page": getattr(el, "page", None)})
        entity_hints = extract_entity_hints(elements_for_hints)
        if entity_hints:
            for chunk in chunks:
                if hasattr(chunk, "entity_hints"):
                    chunk.entity_hints = list(entity_hints)
    except Exception:
        pass  # entity hints are best-effort

    # --- Resolve embedding dimension and ensure collection ---
    embedder = ctx.components.embedder
    dim = embedding_cfg.dimension or embedder.dimension
    vs = ctx.components.vector_store
    sparse = embedder.supports_sparse
    vs.ensure_collection(collection, dense_dim=dim, sparse=sparse)

    # --- Embed ---
    t_embed = time.monotonic()
    texts = [c.text for c in chunks]
    dense_vectors = embedder.embed_passages(texts)
    sparse_vectors: List[Optional[dict]] = [None] * len(chunks)
    if sparse:
        try:
            sparse_raw = embedder.embed_sparse(texts)
            sparse_vectors = sparse_raw
        except Exception as exc:
            logger.warning("Sparse embedding failed, proceeding without sparse: %s", exc)
    embed_ms = (time.monotonic() - t_embed) * 1000

    # --- Store ---
    t_store = time.monotonic()
    batch_points = []
    for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
        section_path = None
        if hasattr(chunk, "hierarchy") and chunk.hierarchy and isinstance(chunk.hierarchy, list):
            section_path = [str(s) for s in chunk.hierarchy if s]
        if not section_path:
            section_str = getattr(chunk, "section", None) or ""
            section_path = [section_str] if section_str else ["_root"]

        payload = ChunkPayload(
            text=chunk.text,
            doc_id=doc_id,
            chunk_id=chunk.chunk_id,
            section_path=section_path,
            element_types=getattr(chunk, "element_types", []) or [],
            page=chunk.page,
            token_count=getattr(chunk, "token_count", None),
            prev_chunk_id=getattr(chunk, "prev_chunk_id", None),
            next_chunk_id=getattr(chunk, "next_chunk_id", None),
            user_id=user_id,
            entity_hints=list(getattr(chunk, "entity_hints", None) or []),
            canonical_type=getattr(chunk, "canonical_type", None),
            source_app=getattr(chunk, "source_app", None),
        )
        batch_points.append({
            "point_id": str(uuid.uuid4()),
            "vector": list(dense_vec) if hasattr(dense_vec, "__iter__") else dense_vec,
            "sparse_vector": sparse_vec,
            "payload": payload,
        })

    vs.upsert_batch(collection=collection, points=batch_points)
    store_ms = (time.monotonic() - t_store) * 1000
    total_ms = (time.monotonic() - t0) * 1000

    cortex_logger.log_ingest(
        app_name=ctx.app_name,
        doc_id=doc_id,
        user_id=user_id,
        strategy=chunking_cfg.strategy,
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
        doc_id, len(chunks), collection, total_ms,
    )

    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "collection": collection,
        "app_name": ctx.app_name,
        "embed_dim": dim,
        "section_paths_sample": [
            c.hierarchy or [getattr(c, "section", "_root")]
            for c in chunks[:3]
        ],
    }
