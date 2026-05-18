from __future__ import annotations

from typing import Any, Dict, Optional


def should_clarify(
    analyzer_result: Dict[str, Any],
    policy: str,
    clarification_reply: bool = False,
) -> bool:
    """
    Decide whether to send a clarification turn based on analyzer result and policy.

    policy: "aggressive" | "balanced" | "never"
    clarification_reply: True when current user message IS a reply to a prior clarification

    Logic:
    - "never" → always False
    - "balanced" → True only if is_ambiguous=True and no previous clarification in this exchange
    - "aggressive" → True if is_ambiguous=True OR query_type=="navigational" and entities is empty
    """
    if policy == "never" or clarification_reply:
        return False
    if not analyzer_result.get("is_ambiguous"):
        if policy == "aggressive":
            # Aggressive: also clarify if navigational query with no entities
            if (analyzer_result.get("query_type") == "navigational" and
                    not analyzer_result.get("entities")):
                return True
        return False
    return True


def build_clarification_context(
    analyzer_result: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    """Build the context dict stored on the thread when a clarification is pending."""
    return {
        "original_query": query,
        "clarifying_question": analyzer_result.get("clarifying_question"),
        "analyzer_result": analyzer_result,
    }
