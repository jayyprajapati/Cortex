from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_clarification_pending(thread: Dict[str, Any]) -> bool:
    """Check if this thread is waiting for a clarification reply."""
    return bool(thread.get("clarification_pending", False))


def get_clarification_context(thread: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the stored clarification context dict (original query + clarifying question)."""
    ctx_raw = thread.get("clarification_context")
    if not ctx_raw:
        return None
    if isinstance(ctx_raw, dict):
        return ctx_raw
    try:
        return json.loads(str(ctx_raw))
    except Exception:
        return None


def mark_clarification_pending(thread_id: str, context: Dict[str, Any]) -> None:
    """Mark a thread as waiting for clarification and store the context."""
    try:
        from app.threads.store import update_thread_meta
        update_thread_meta(thread_id, {
            "clarification_pending": True,
            "clarification_context": json.dumps(context),
        })
    except Exception as exc:
        logger.warning("Failed to mark clarification pending for thread %s: %s", thread_id, exc)


def clear_clarification_pending(thread_id: str) -> None:
    """Clear the clarification pending flag after the user replies."""
    try:
        from app.threads.store import update_thread_meta
        update_thread_meta(thread_id, {
            "clarification_pending": False,
            "clarification_context": None,
        })
    except Exception as exc:
        logger.warning("Failed to clear clarification pending for thread %s: %s", thread_id, exc)
