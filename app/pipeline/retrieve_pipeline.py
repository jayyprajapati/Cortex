from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.context import ExecutionContext
from app.observability.logger import cortex_logger
from app.reranker.reranker import rerank
from app.retrieval.search import retrieve

logger = logging.getLogger(__name__)


def retrieve_and_rerank(ctx: ExecutionContext, query: str) -> dict:
    """
    Full retrieval + reranking pipeline.

    Returns:
        {chunks: List[dict], is_vague: bool, doc_ids: List[str], clarification: str|None}
    """
    t0 = time.monotonic()

    retrieval_result = retrieve(ctx, query)
    chunks: List[dict] = retrieval_result.get("chunks", [])

    # Propagate early-exit conditions without reranking
    if not chunks or retrieval_result.get("clarification"):
        return retrieval_result

    reranking_cfg = ctx.registry.reranking
    rerank_ms = 0.0
    if reranking_cfg.enabled:
        t_rerank = time.monotonic()
        chunks = rerank(query, chunks, reranking_cfg)
        rerank_ms = (time.monotonic() - t_rerank) * 1000

        cortex_logger.log_rerank(
            app_name=ctx.app_name,
            model=reranking_cfg.model,
            candidates=len(retrieval_result.get("chunks", [])),
            selected=len(chunks),
            latency_ms=rerank_ms,
        )

    total_ms = (time.monotonic() - t0) * 1000

    retrieval_scores = [c.get("score", 0.0) for c in chunks]
    rerank_scores = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]

    cortex_logger.log_retrieve(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        query_len=len(query),
        chunk_count=len(chunks),
        rerank_enabled=reranking_cfg.enabled,
        avg_score=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        avg_rerank=sum(rerank_scores) / len(rerank_scores) if rerank_scores else None,
        rerank_latency_ms=rerank_ms,
        total_latency_ms=total_ms,
    )

    return {
        "chunks": chunks,
        "is_vague": retrieval_result.get("is_vague", False),
        "doc_ids": retrieval_result.get("doc_ids", []),
        "clarification": retrieval_result.get("clarification"),
    }
