"""Rolling LLM summary of older conversation turns.

When a thread's message count grows past a threshold, older messages are
compressed into a short summary so we don't have to replay the entire history
in every prompt. The summary preserves named entities, dates, decisions, and
unresolved questions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Window of most-recent messages we always keep verbatim in the prompt.
# Anything older than this gets folded into the summary.
KEEP_RECENT = 10

# Trigger summarization once total messages exceeds this number AND we have
# at least KEEP_RECENT + SUMMARIZE_AFTER unsummarized messages to compress.
SUMMARIZE_AFTER = 10

_SUMMARY_PROMPT = """\
You are a conversation summarizer. Compress the conversation below into a 3-5 sentence summary.

Requirements:
- Preserve named entities (people, companies, documents, dates, numbers).
- Note any decisions reached.
- Note any unresolved questions or pending follow-ups.
- Use third-person ("the user asked...", "the assistant answered...").

If a prior summary is provided, fold the new conversation into it (don't list things twice).

PRIOR SUMMARY:
{prior_summary}

CONVERSATION TO ABSORB:
{conversation}

UPDATED SUMMARY:"""


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        # Truncate very long single messages to keep the summary prompt bounded.
        if len(content) > 1500:
            content = content[:1500] + "…"
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


def summarize_old_turns(
    older_messages: List[Dict[str, Any]],
    prior_summary: str | None,
    llm,
) -> str:
    """Produce a rolling summary of older_messages, optionally folding in prior_summary."""
    if not older_messages:
        return prior_summary or ""

    prompt = _SUMMARY_PROMPT.format(
        prior_summary=prior_summary.strip() if prior_summary else "(none yet)",
        conversation=_format_conversation(older_messages),
    )
    try:
        result = llm.generate(prompt, temperature=0.1)
        return str(result or "").strip()
    except Exception as exc:
        logger.warning("Thread summarization failed: %s", exc)
        # Fall back to the prior summary; don't drop history because of a transient LLM error.
        return prior_summary or ""
