from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.context import ExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_count: int = 0
    reranked_count: int = 0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    confidence: str = "low"
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    doc_ids: List[str] = field(default_factory=list)
    rewrites: List[str] = field(default_factory=list)
    is_vague: bool = False


def run_retrieval(
    ctx: ExecutionContext,
    query: str,
    history_summary: Optional[str] = None,
    last_assistant_msg: Optional[str] = None,
    clarification_reply: bool = False,
) -> RetrievalResult:
    """
    Full retrieval pipeline:
    1. Query analysis (ambiguity check + rewrites)
    2. Clarification gate (if ambiguous + policy allows)
    3. Query expansion (original + rewrites + optional HyDE)
    4. Hybrid vector search
    5. Reranking
    6. Neighbor expansion
    7. Confidence scoring

    Returns RetrievalResult.
    """
    t0 = time.monotonic()
    retrieval_cfg = ctx.registry.retrieval
    conv_cfg = ctx.registry.conversation
    policy = conv_cfg.clarification_policy

    # Step 1: Query analysis
    analyzer_result: Dict[str, Any] = {"is_ambiguous": False, "clarifying_question": None, "rewrites": [], "entities": [], "query_type": "factual"}
    if conv_cfg.use_query_analyzer and not clarification_reply:
        from app.retrieval.query_analyzer import analyze_query
        analyzer_result = analyze_query(ctx, query, last_assistant_msg=last_assistant_msg)

    rewrites = [r for r in (analyzer_result.get("rewrites") or []) if r and r.strip() and r != query]

    # Step 2: Clarification gate
    if (
        analyzer_result.get("is_ambiguous")
        and policy != "never"
        and not clarification_reply
    ):
        clarifying_q = analyzer_result.get("clarifying_question") or "Could you provide more context?"
        return RetrievalResult(
            needs_clarification=True,
            clarifying_question=clarifying_q,
            is_vague=True,
        )

    # Step 3: Query expansion
    queries = [query] + rewrites
    if retrieval_cfg.hyde and not rewrites:
        from app.retrieval.hyde import generate_hypothetical_doc
        hyde_text = generate_hypothetical_doc(ctx, query)
        if hyde_text and hyde_text != query:
            queries.append(hyde_text)

    # Step 4: Vector search
    vs = ctx.components.vector_store if ctx.components else None
    embedder = ctx.components.embedder if ctx.components else None

    if vs is None or embedder is None:
        logger.warning("No vector store or embedder in ctx.components — returning empty result")
        return RetrievalResult()

    # Check if user has any documents
    user_docs = vs.list_docs(ctx.collection, ctx.user_id)
    if not user_docs:
        return RetrievalResult(
            needs_clarification=False,
            is_vague=False,
        )

    requested_doc_ids = ctx.doc_ids or None
    candidate_cap = retrieval_cfg.top_k * 3 if not ctx.registry.reranking.enabled else ctx.registry.reranking.candidate_cap
    candidate_cap = max(candidate_cap, retrieval_cfg.top_k * 2)

    all_chunks: List[Dict] = []
    for q in queries:
        try:
            query_vec = embedder.embed_query(q)
            sparse_vec = None
            if embedder.supports_sparse:
                try:
                    sparse_results = embedder.embed_sparse([q])
                    sparse_vec = sparse_results[0] if sparse_results else None
                except Exception:
                    sparse_vec = None

            chunks = vs.search_hybrid(
                collection=ctx.collection,
                query_vector=query_vec,
                sparse_vector=sparse_vec,
                user_id=ctx.user_id,
                top_k=candidate_cap,
                doc_ids=requested_doc_ids,
                fusion=retrieval_cfg.fusion,
                alpha=retrieval_cfg.alpha,
                metadata_filter=retrieval_cfg.metadata_filter,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("Search failed for query variant %r: %s", q[:60], exc)

    # Deduplicate
    seen: Dict[str, Dict] = {}
    for c in all_chunks:
        doc_id = str(c.get("doc_id") or "")
        chunk_id = c.get("chunk_id")
        key = f"{doc_id}:{chunk_id}" if chunk_id is not None else f"{doc_id}:{str(c.get('text') or '')[:80]}"
        if key not in seen or c.get("score", 0.0) > seen[key].get("score", 0.0):
            seen[key] = c
    merged = list(seen.values())
    retrieved_count = len(merged)

    if not merged:
        return RetrievalResult(retrieved_count=0)

    # Step 5: Reranking
    reranked_count = len(merged)
    rerank_ms = 0.0
    reranking_cfg = ctx.registry.reranking

    if reranking_cfg.enabled and ctx.components and ctx.components.reranker:
        t_rerank = time.monotonic()
        merged = ctx.components.reranker.rerank(
            query=query,
            chunks=merged,
            top_k=reranking_cfg.top_k,
            candidate_cap=reranking_cfg.candidate_cap,
        )
        rerank_ms = (time.monotonic() - t_rerank) * 1000
        reranked_count = len(merged)
    else:
        merged.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        merged = merged[:retrieval_cfg.top_k]
        reranked_count = len(merged)

    # Step 6: Neighbor expansion
    if retrieval_cfg.expand_neighbors:
        from app.retrieval.neighbor_expansion import expand_neighbors
        merged = expand_neighbors(ctx, merged, budget_tokens=retrieval_cfg.neighbor_budget_tokens)

    # Step 7: Confidence
    from app.retrieval.confidence import compute_confidence
    confidence = compute_confidence(merged, min_score=retrieval_cfg.confidence_min_score)

    total_ms = (time.monotonic() - t0) * 1000
    discovered_doc_ids = sorted({str(c["doc_id"]) for c in merged if c.get("doc_id")})

    return RetrievalResult(
        chunks=merged,
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        rerank_latency_ms=round(rerank_ms, 1),
        total_latency_ms=round(total_ms, 1),
        confidence=confidence,
        needs_clarification=False,
        doc_ids=discovered_doc_ids,
        rewrites=rewrites,
        is_vague=False,
    )
