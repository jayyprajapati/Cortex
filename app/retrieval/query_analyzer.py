from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.context import ExecutionContext

logger = logging.getLogger(__name__)

_ANALYZER_PROMPT = """Analyze this search query for a RAG (Retrieval-Augmented Generation) system and return ONLY valid JSON.

Context: The user is querying their own uploaded documents. References like "this document", "the file", "the resume", "it", "the one I uploaded", "the document" always refer to the user's uploaded documents and are NEVER ambiguous — do NOT set is_ambiguous=true for these.

Query: {query}
{history_context}

Return JSON with exactly these fields:
{{
  "is_ambiguous": <bool — true ONLY if the query has absolutely no searchable intent, like a single random word or a pure meta-question about the system itself>,
  "clarifying_question": <str or null — if ambiguous, a specific clarifying question>,
  "entities": <list of str — named entities in the query>,
  "query_type": <"factual" | "conceptual" | "navigational" | "comparison">,
  "rewrites": <list of 1-3 str — alternative phrasings for the same intent, empty if not useful>
}}

Rules:
- Document-reference queries ("what is this about?", "summarize it", "what does the file say?") are NOT ambiguous — the document is the user's upload
- Only set is_ambiguous=true for queries that have zero searchable content (e.g. a single stop-word, or "hello")
- rewrites should be genuinely different phrasings, not paraphrases
- Return ONLY the JSON object, no prose"""


def analyze_query(
    ctx: ExecutionContext,
    query: str,
    last_assistant_msg: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a lightweight LLM call to analyze the query.
    Returns a dict with: is_ambiguous, clarifying_question, entities, query_type, rewrites.
    Falls back to a safe default dict on any error.
    """
    default = {
        "is_ambiguous": False,
        "clarifying_question": None,
        "entities": [],
        "query_type": "factual",
        "rewrites": [],
    }

    if not ctx.registry.conversation.use_query_analyzer:
        return default

    history_context = ""
    if last_assistant_msg:
        history_context = f"Previous assistant message: {last_assistant_msg[:300]}"

    prompt = _ANALYZER_PROMPT.format(query=query, history_context=history_context)

    try:
        from app.llm.factory import get_llm
        llm = get_llm(ctx.llm_config)
        analyzer_temp = getattr(ctx.registry.conversation, "analyzer_temperature", 0.0)
        raw = llm.generate(prompt, temperature=analyzer_temp)
        # Parse JSON from response
        raw = str(raw or "").strip()
        # Strip code fences if present
        import re
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL)
        if fence:
            raw = fence.group(1)
        # Find JSON object
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            result = json.loads(raw[s:e+1])
            # Ensure all expected keys are present
            for k, v in default.items():
                if k not in result:
                    result[k] = v
            return result
    except Exception as exc:
        logger.debug("Query analyzer failed (non-fatal): %s", exc)

    return default
