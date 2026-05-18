"""LLM-based query rewriter for higher retrieval recall.

Given a user question, produce 1-3 alternative phrasings that:
  - expand synonyms ("developer" -> "software engineer", "programmer", "SDE")
  - resolve pronouns/anaphora using prior conversation context
  - make implicit subjects explicit on vague queries ("tell me more" -> "...")

Cached in-process by (query, history_summary_hash) so repeated queries are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a retrieval-query rewriter. Your job is to maximize recall over a vector
database of document chunks (resumes, papers, manuals).

Given a user question and optional conversation context, output 1-3 alternative
phrasings that would help find the relevant passages.

Rules:
- Expand synonyms (e.g., "developer" → "software engineer", "programmer").
- Resolve pronouns and anaphora using the conversation context.
- If the question is vague (e.g., "tell me more"), make the implicit subject explicit.
- Each rewrite should be a complete, self-contained query.
- Keep each rewrite under 30 words.

Output ONLY valid JSON in this exact shape (no markdown fences, no prose):
{{"rewrites": ["rewrite 1", "rewrite 2"], "rationale": "one short sentence"}}

Conversation context: {history}
User question: {query}

JSON Output:"""


@dataclass
class RewrittenQuery:
    rewrites: List[str]
    rationale: str

    @property
    def all_queries(self) -> List[str]:
        """Original-style accessor: caller usually uses [original, *rewrites]."""
        return list(self.rewrites)


# Simple thread-safe LRU cache keyed by (query, summary_hash)
_CACHE_MAX = 256
_cache: "OrderedDict[str, RewrittenQuery]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str, history_summary: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8", errors="replace"))
    h.update(b"||")
    h.update((history_summary or "").encode("utf-8", errors="replace"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[RewrittenQuery]:
    with _cache_lock:
        if key not in _cache:
            return None
        _cache.move_to_end(key)
        return _cache[key]


def _cache_put(key: str, value: RewrittenQuery) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_rewrites(raw: str) -> RewrittenQuery:
    raw = (raw or "").strip()
    # Strip markdown fences
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    # Best-effort find a JSON object
    if not raw.startswith("{"):
        m = _JSON_BLOCK_RE.search(raw)
        if m:
            raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"rewriter returned non-JSON: {raw[:200]!r}") from exc

    rewrites_raw = data.get("rewrites") or []
    if not isinstance(rewrites_raw, list):
        rewrites_raw = []
    rewrites: List[str] = []
    for r in rewrites_raw:
        s = str(r or "").strip()
        if s and len(rewrites) < 3:
            rewrites.append(s)

    return RewrittenQuery(
        rewrites=rewrites,
        rationale=str(data.get("rationale") or "").strip(),
    )


def rewrite_query(
    query: str,
    history_summary: Optional[str],
    llm,
) -> RewrittenQuery:
    """Produce 1-3 retrieval-friendly rewrites of `query`.

    Returns a RewrittenQuery with rewrites=[] if the LLM call fails — callers
    should fall back to the original query alone.
    """
    q = (query or "").strip()
    if not q:
        return RewrittenQuery(rewrites=[], rationale="")

    key = _cache_key(q, history_summary)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    prompt = _REWRITE_PROMPT.format(
        query=q,
        history=(history_summary or "(none)").strip(),
    )

    try:
        raw = llm.generate(prompt, temperature=0.1)
    except Exception as exc:
        logger.warning("Query rewrite LLM call failed: %s", exc)
        return RewrittenQuery(rewrites=[], rationale="")

    try:
        parsed = _parse_rewrites(str(raw or ""))
    except ValueError as exc:
        logger.warning("Query rewrite parse failed: %s", exc)
        parsed = RewrittenQuery(rewrites=[], rationale="")

    _cache_put(key, parsed)
    return parsed
