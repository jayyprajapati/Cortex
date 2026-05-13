from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tuneable constants — kept in sync with app/threads/summarize.py (KEEP_RECENT=10, SUMMARIZE_AFTER=10).
KEEP_RECENT = 10
SUMMARY_THRESHOLD = KEEP_RECENT + 10  # == 20; matches SUMMARIZE_AFTER trigger in summarize.py


def build_history_window(
    messages: List[Dict[str, Any]],
    summary: Optional[str],
    max_turns: int = KEEP_RECENT,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Given a list of stored messages (newest last) and an optional rolling summary,
    return (recent_window, summary) where:
    - recent_window is the last `max_turns` messages as {role, content} dicts
    - summary is passed through unchanged

    The window is oldest-first for prompt construction.
    """
    if not messages:
        return [], summary

    # Take the tail (most recent turns)
    tail = messages[-max_turns:] if len(messages) > max_turns else messages
    window = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in tail
        if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
    ]
    return window, summary


def extract_last_assistant_msg(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Return the most recent assistant message content, or None."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = str(m.get("content") or "").strip()
            if content:
                return content[:500]
    return None


def should_summarize(total_message_count: int, summarized_up_to: int) -> bool:
    """Return True when enough new messages have accumulated to trigger summarization."""
    unsummarized = total_message_count - summarized_up_to - KEEP_RECENT
    return unsummarized >= SUMMARY_THRESHOLD
