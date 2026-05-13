from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def reindex_app(
    app_name: str,
    user_id: str,
    source_dir: str,
    drop_first: bool = True,
) -> Dict[str, Any]:
    """
    Drop the app's Qdrant collection (if drop_first=True) and re-ingest
    every supported file from source_dir.
    Returns per-file stats.
    """
    from app.registry.service import build_execution_context
    from app.pipeline.ingest_pipeline import ingest_document, resolve_doc_id

    ctx = build_execution_context(app_name=app_name, user_id=user_id)
    vs = ctx.components.vector_store
    collection = ctx.collection

    if drop_first:
        try:
            vs.drop_collection(collection)
            logger.info("Dropped collection '%s'", collection)
        except Exception as exc:
            logger.warning("Drop collection '%s' failed (may not exist): %s", collection, exc)

    if not os.path.isdir(source_dir):
        raise ValueError(f"source_dir {source_dir!r} is not a directory or does not exist")

    # Collect files with supported extensions
    SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".markdown", ".txt"}
    file_paths: List[str] = []
    for root, _dirs, files in os.walk(source_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                file_paths.append(os.path.join(root, fname))

    if not file_paths:
        return {
            "app_name": app_name,
            "collection": collection,
            "dropped": drop_first,
            "files_processed": 0,
            "total_chunks": 0,
            "results": [],
        }

    results: List[Dict[str, Any]] = []
    total_chunks = 0

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        doc_id = resolve_doc_id(os.path.splitext(fname)[0])
        try:
            result = ingest_document(ctx, fpath, doc_id)
            chunks = result.get("chunk_count", 0)
            total_chunks += chunks
            results.append({
                "file": fname,
                "doc_id": doc_id,
                "chunk_count": chunks,
                "status": "ok",
            })
            logger.info("Reindexed %s: %d chunks", fname, chunks)
        except Exception as exc:
            results.append({
                "file": fname,
                "doc_id": doc_id,
                "chunk_count": 0,
                "status": "error",
                "error": str(exc),
            })
            logger.error("Reindex failed for %s: %s", fname, exc)

    return {
        "app_name": app_name,
        "collection": collection,
        "dropped": drop_first,
        "files_processed": len(file_paths),
        "total_chunks": total_chunks,
        "results": results,
    }
