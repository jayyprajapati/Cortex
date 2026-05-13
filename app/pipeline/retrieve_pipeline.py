from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.context import ExecutionContext

logger = logging.getLogger(__name__)


def retrieve_and_rerank(
    ctx: ExecutionContext,
    query: str,
    history_summary: Optional[str] = None,
    clarification_reply: bool = False,
) -> dict:
    """
    Thin wrapper over app.retrieval.pipeline.run_retrieval().
    Returns a dict matching the legacy shape expected by generate_pipeline.py:
    {chunks, retrieved_count, reranked_count, rerank_latency_ms, is_vague, doc_ids,
     needs_clarification, clarifying_question, rewrites}
    """
    from app.retrieval.pipeline import run_retrieval

    last_assistant = None

    result = run_retrieval(
        ctx=ctx,
        query=query,
        history_summary=history_summary,
        last_assistant_msg=last_assistant,
        clarification_reply=clarification_reply,
    )

    return {
        "chunks": result.chunks,
        "retrieved_count": result.retrieved_count,
        "reranked_count": result.reranked_count,
        "rerank_latency_ms": result.rerank_latency_ms,
        "is_vague": result.is_vague,
        "doc_ids": result.doc_ids,
        "needs_clarification": result.needs_clarification,
        "clarifying_question": result.clarifying_question,
        "rewrites": result.rewrites,
        # Legacy compat: generate_pipeline.py checks for "clarification" key
        "clarification": result.clarifying_question if result.needs_clarification else None,
    }
