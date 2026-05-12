from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.context import ExecutionContext
from app.llm.factory import get_llm
from app.observability.logger import cortex_logger
from app.reranker.reranker import rerank
from app.retrieval.rewrite import rewrite_query
from app.retrieval.search import retrieve

logger = logging.getLogger(__name__)


def _chunk_key(c: Dict[str, Any]) -> tuple:
    """Stable dedup key across multi-query retrievals."""
    doc_id = str(c.get("doc_id") or "")
    chunk_id = c.get("chunk_id")
    if chunk_id is not None:
        return (doc_id, chunk_id)
    section = str(c.get("section") or "")
    page = c.get("page")
    text_fingerprint = " ".join(str(c.get("text") or "").split())[:120]
    return (doc_id, section, page, text_fingerprint)


def _union_chunks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge multi-query retrieval results, deduplicate, keep the best per-key score."""
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for r in results:
        for c in r.get("chunks", []):
            k = _chunk_key(c)
            if k not in by_key or c.get("score", 0.0) > by_key[k].get("score", 0.0):
                by_key[k] = c
    return list(by_key.values())


def retrieve_and_rerank(
    ctx: ExecutionContext,
    query: str,
    history_summary: Optional[str] = None,
) -> dict:
    """
    Full retrieval + reranking pipeline.

    When `retrieval.query_rewrite` is enabled, the original query is run alongside
    1-3 LLM-generated rewrites; results are unioned, deduplicated, and reranked
    against the ORIGINAL query.

    Returns:
        {chunks, retrieved_count, reranked_count, rerank_latency_ms,
         is_vague, doc_ids, clarification, rewrites?}
    """
    t0 = time.monotonic()
    retrieval_cfg = ctx.registry.retrieval

    # ── Optional: multi-query expansion via LLM rewrite ──
    rewrites: List[str] = []
    if retrieval_cfg.query_rewrite:
        try:
            llm = get_llm(ctx.llm_config)
            rewritten = rewrite_query(query, history_summary, llm)
            rewrites = rewritten.rewrites
        except Exception as exc:
            logger.warning("Query rewrite stage failed, falling back to original: %s", exc)
            rewrites = []

    queries = [query] + [r for r in rewrites if r and r != query]

    # ── Retrieve from each query variant ──
    per_query_results: List[Dict[str, Any]] = []
    first_clarification: Optional[str] = None
    for q in queries:
        r = retrieve(ctx, q)
        if r.get("clarification") and first_clarification is None:
            first_clarification = r.get("clarification")
        per_query_results.append(r)

    # If no docs are even ingested, propagate the clarification
    if first_clarification and not any(r.get("chunks") for r in per_query_results):
        first = per_query_results[0]
        first["rewrites"] = rewrites
        return first

    # Union + dedupe across queries
    merged_chunks: List[Dict[str, Any]] = _union_chunks(per_query_results)
    retrieved_count = len(merged_chunks)

    if not merged_chunks:
        first = per_query_results[0]
        first["rewrites"] = rewrites
        return first

    reranking_cfg = ctx.registry.reranking
    rerank_ms = 0.0
    if reranking_cfg.enabled:
        t_rerank = time.monotonic()
        # Always rerank against the original user query (not rewrites)
        merged_chunks = rerank(query, merged_chunks, reranking_cfg)
        rerank_ms = (time.monotonic() - t_rerank) * 1000

        cortex_logger.log_rerank(
            app_name=ctx.app_name,
            model=reranking_cfg.model,
            candidates=retrieved_count,
            selected=len(merged_chunks),
            latency_ms=rerank_ms,
        )
    else:
        merged_chunks = sorted(merged_chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        merged_chunks = merged_chunks[: retrieval_cfg.top_k]

    total_ms = (time.monotonic() - t0) * 1000

    retrieval_scores = [c.get("score", 0.0) for c in merged_chunks]
    rerank_scores = [c["rerank_score"] for c in merged_chunks if c.get("rerank_score") is not None]

    cortex_logger.log_retrieve(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        query_len=len(query),
        chunk_count=len(merged_chunks),
        rerank_enabled=reranking_cfg.enabled,
        avg_score=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        avg_rerank=sum(rerank_scores) / len(rerank_scores) if rerank_scores else None,
        rerank_latency_ms=rerank_ms,
        total_latency_ms=total_ms,
    )

    # Doc id set — discovered across all query variants
    discovered_doc_ids = sorted({
        str(c["doc_id"]) for c in merged_chunks if c.get("doc_id")
    })

    return {
        "chunks": merged_chunks,
        "retrieved_count": retrieved_count,
        "reranked_count": len(merged_chunks),
        "rerank_latency_ms": round(rerank_ms, 1),
        "is_vague": per_query_results[0].get("is_vague", False),
        "doc_ids": discovered_doc_ids or per_query_results[0].get("doc_ids", []),
        "clarification": first_clarification if not merged_chunks else None,
        "rewrites": rewrites,
    }
