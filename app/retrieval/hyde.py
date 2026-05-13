from __future__ import annotations

import logging
from typing import List

from app.context import ExecutionContext

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """Write a short passage (2-3 sentences) that would be a perfect answer to this question.
Write it as if it were content from a document, not as a direct answer.

Question: {query}

Passage:"""


def generate_hypothetical_doc(ctx: ExecutionContext, query: str) -> str:
    """
    Generate a hypothetical document passage for the query.
    Returns the hypothetical text, or the original query on failure.
    """
    try:
        from app.llm.factory import get_llm
        llm = get_llm(ctx.llm_config)
        hypothesis = llm.generate(_HYDE_PROMPT.format(query=query), temperature=0.7)
        return str(hypothesis or "").strip() or query
    except Exception as exc:
        logger.debug("HyDE generation failed (non-fatal): %s", exc)
        return query
