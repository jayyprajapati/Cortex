from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.context import EffectiveGenerationConfig, ExecutionContext
from app.llm.factory import get_llm
from app.observability.logger import cortex_logger
from app.pipeline.retrieve_pipeline import retrieve_and_rerank

logger = logging.getLogger(__name__)

_NO_CONTEXT = "No relevant information found."


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

def _extract_json_payload(raw: Any) -> Any:
    if raw is None:
        raise ValueError("Model output is empty")

    stripped = str(raw).strip()
    if not stripped:
        raise ValueError("Model output is empty after stripping whitespace")

    # Strip markdown code fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    candidates = [stripped]
    obj_s, obj_e = stripped.find("{"), stripped.rfind("}")
    if obj_s != -1 and obj_e > obj_s:
        candidates.append(stripped[obj_s : obj_e + 1])
    arr_s, arr_e = stripped.find("["), stripped.rfind("]")
    if arr_s != -1 and arr_e > arr_s:
        candidates.append(stripped[arr_s : arr_e + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Model output is not valid JSON. First 300 chars: {stripped[:300]!r}"
    )


def _type_ok(value: Any, expected: str) -> bool:
    checks: Dict[str, Any] = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    fn = checks.get(expected)
    return fn(value) if fn else True


def _validate_schema(data: Any, schema: Dict, path: str = "$") -> None:
    if not schema:
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_type_ok(data, t) for t in types):
            raise ValueError(
                f"Type mismatch at {path}: expected {expected_type}, "
                f"got {type(data).__name__}"
            )

    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                raise ValueError(f"Missing required field at {path}.{req}")
        for key, val in data.items():
            prop_schema = schema.get("properties", {}).get(key)
            if prop_schema:
                _validate_schema(val, prop_schema, f"{path}.{key}")

    if isinstance(data, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                _validate_schema(item, item_schema, f"{path}[{i}]")


def _normalize_schema(schema: Optional[Dict]) -> Optional[Dict]:
    """Convert shorthand {field: type} mapping to JSON Schema object."""
    if schema is None or "type" in schema:
        return schema

    type_map = {
        "string": "string", "number": "number", "array": "array",
        "object": "object", "boolean": "boolean", "integer": "integer", "null": "null",
    }
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for field_name, field_type in schema.items():
        t = type_map.get(str(field_type).strip().lower(), "string")
        properties[field_name] = {"type": t}
        required.append(field_name)

    return {"type": "object", "required": required, "properties": properties}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

STRICT_REFUSAL = "I couldn't find that in the document."

_STRICT_RULES_MD = """=== ABSOLUTE RULES (you MUST follow) ===
1. Answer ONLY using facts that appear in the CONTEXT below.
2. Every factual claim must cite source numbers like [1] or [2, 3].
3. If the CONTEXT does not contain enough information, output EXACTLY this and nothing else:
   {refusal}
   Do not guess. Do not use general knowledge. Do not invent names, dates, or facts.
4. Quote names, dates, numbers, and proper nouns VERBATIM from the context. Never paraphrase or alter a name.
5. Do not draw on prior knowledge or training data.

=== ANSWER FORMAT ===
Produce your answer in this exact markdown structure:

**Answer:** <one short paragraph, direct, with inline [N] citations>

**Reasoning:** <1-2 sentences explaining which source supports the answer>

**Sources:** [1] [2] ..."""


_TRUTHFUL_CLAUSE = (
    "\n\nTRUTHFUL OUTPUT: You may rephrase, restructure, and emphasize content, "
    "but every concrete fact (company name, title, date, metric, project name, skill, person name) "
    "must originate from the provided context. Do not invent facts or attribute things to entities "
    "that are not present in the source.\n"
)


def _build_prompt_basic(query: str, context: str, gen: EffectiveGenerationConfig) -> str:
    system = gen.system_prompt.strip()
    q = str(query or "").strip()
    ctx = str(context or "").strip()

    if gen.response_type == "json":
        return (
            f"{system}\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- Do not include markdown, code fences, or prose outside JSON.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Input:\n{q}\n\n"
            "JSON Output:"
        )

    return f"{system}\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"


def _format_history_block(history: Optional[List[Dict[str, Any]]], summary: Optional[str]) -> str:
    parts: List[str] = []
    if summary and summary.strip():
        parts.append(f"=== CONVERSATION SUMMARY ===\n{summary.strip()}")
    if history:
        lines: List[str] = []
        for m in history:
            role = str(m.get("role") or "").strip().lower()
            content = str(m.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            lines.append(f"{role.capitalize()}: {content}")
        if lines:
            parts.append("=== RECENT TURNS ===\n" + "\n\n".join(lines))
    return "\n\n".join(parts)


def _build_prompt_strict_markdown(
    query: str,
    context: str,
    gen: EffectiveGenerationConfig,
    history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> str:
    system = gen.system_prompt.strip()
    q = str(query or "").strip()
    ctx = str(context or "").strip()
    rules = _STRICT_RULES_MD.format(refusal=STRICT_REFUSAL)
    history_block = _format_history_block(history, summary)

    sections: List[str] = [system]
    if history_block:
        sections.append(history_block)
    sections.append(rules)
    sections.append(f"=== CONTEXT ===\n{ctx}")
    sections.append(f"=== QUESTION ===\n{q}")
    sections.append("Answer:")
    return "\n\n".join(sections)


def _build_prompt_strict_json(
    query: str,
    context: str,
    gen: EffectiveGenerationConfig,
    history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> str:
    system = gen.system_prompt.strip()
    q = str(query or "").strip()
    ctx = str(context or "").strip()
    history_block = _format_history_block(history, summary)

    sections: List[str] = [system]
    if history_block:
        sections.append(history_block)
    sections.append(
        "Rules:\n"
        "- Return ONLY valid JSON.\n"
        "- Do not include markdown, code fences, or prose outside JSON.\n"
        "- Answer ONLY from the CONTEXT. Do not invent names, dates, numbers, or facts.\n"
        "- Quote names and proper nouns VERBATIM from the context."
    )
    sections.append(f"Context:\n{ctx}")
    sections.append(f"Input:\n{q}")
    sections.append("JSON Output:")
    return "\n\n".join(sections)


def _build_prompt_truthful(query: str, context: str, gen: EffectiveGenerationConfig) -> str:
    """Like the basic prompt but with a truthfulness clause prepended to the system prompt."""
    augmented = EffectiveGenerationConfig(
        system_prompt=gen.system_prompt.strip() + _TRUTHFUL_CLAUSE,
        response_type=gen.response_type,
        schema=gen.schema,
        temperature=gen.temperature,
        strict=gen.strict,
        max_retries=gen.max_retries,
        grounding_mode=gen.grounding_mode,
    )
    return _build_prompt_basic(query, context, augmented)


def _build_prompt(
    query: str,
    context: str,
    gen: EffectiveGenerationConfig,
    history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> str:
    """Dispatch on grounding_mode + response_type. history/summary only apply to strict mode."""
    mode = getattr(gen, "grounding_mode", "off") or "off"
    if mode == "strict":
        if gen.response_type == "json":
            return _build_prompt_strict_json(query, context, gen, history=history, summary=summary)
        return _build_prompt_strict_markdown(query, context, gen, history=history, summary=summary)
    if mode == "truthful":
        return _build_prompt_truthful(query, context, gen)
    return _build_prompt_basic(query, context, gen)


def _build_context_str(chunks: List[dict], max_context_tokens: int = 4000) -> str:
    # Drop lowest-ranked chunks (end of list) until the context fits in the budget.
    # Token estimate: len(text) // 4 (fast, no tiktoken dependency).
    budget = max_context_tokens
    kept: List[dict] = []
    for chunk in chunks:
        cost = len(str(chunk.get("text") or "")) // 4
        if cost > budget and kept:
            break
        kept.append(chunk)
        budget -= cost

    blocks = []
    for i, chunk in enumerate(kept):
        section = str(chunk.get("section") or "Untitled").strip()
        page = chunk.get("page")
        page_label = page if page is not None else "N/A"
        blocks.append(
            f"[Source {i + 1}. Section: {section}. Page: {page_label}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _dedupe_chunks(chunks: List[dict]) -> List[dict]:
    seen: set = set()
    result: List[dict] = []
    for chunk in chunks:
        key = (
            str(chunk.get("doc_id") or "").strip(),
            chunk.get("page"),
            str(chunk.get("section") or "").strip(),
            " ".join(str(chunk.get("text") or "").split()),
        )
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer_text: str, sources: List[dict]) -> List[Dict[str, Any]]:
    """Parse [N] / [N, M] markers from a markdown answer and resolve to sources."""
    if not answer_text or not sources:
        return []

    seen_indices: List[int] = []
    for match in _CITATION_RE.finditer(answer_text):
        for piece in match.group(1).split(","):
            try:
                idx = int(piece.strip())
            except ValueError:
                continue
            if 1 <= idx <= len(sources) and idx not in seen_indices:
                seen_indices.append(idx)

    citations: List[Dict[str, Any]] = []
    for idx in seen_indices:
        src = sources[idx - 1]
        citations.append({
            "index": idx,
            "section": src.get("section"),
            "page": src.get("page"),
            "text": src.get("text"),
        })
    return citations


def _build_public_sources(chunks: List[dict]) -> List[dict]:
    sources = []
    for chunk in chunks:
        item: Dict[str, Any] = {
            "section": str(chunk.get("section") or "Untitled").strip(),
            "page": chunk.get("page"),
        }
        text = str(chunk.get("text") or "").strip()
        if text:
            item["text"] = text
        if chunk.get("score") is not None:
            item["score"] = chunk["score"]
        if chunk.get("rerank_score") is not None:
            item["rerank_score"] = chunk["rerank_score"]
        if chunk.get("hierarchy"):
            item["hierarchy"] = chunk["hierarchy"]
        sources.append(item)
    return sources


# ---------------------------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------------------------

# Capitalized tokens that frequently appear in generated prose but are not
# real proper nouns — filter them out of the verification candidate set.
_COMMON_NON_PROPER = {
    "The", "This", "That", "These", "Those", "It", "He", "She", "They", "We",
    "I", "You", "There", "Here", "Then", "Now", "Today", "Yesterday", "Tomorrow",
    "Answer", "Reasoning", "Sources", "Source", "Question", "Context", "Note",
    "Yes", "No", "Output", "Input", "JSON", "Rules", "Format", "Section",
    "Page", "PDF", "URL", "ATS",
}

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d{2,}\b")


def _flatten_to_text(value: Any) -> str:
    """Concatenate every string leaf in a nested value into one verification haystack."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(_flatten_to_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_to_text(v) for v in value)
    return str(value)


def _extract_verifiable_tokens(text: str) -> Dict[str, List[str]]:
    """Pull out proper nouns and numeric tokens that need to be grounded in the source."""
    proper: List[str] = []
    for match in _PROPER_NOUN_RE.finditer(text or ""):
        candidate = match.group(1).strip()
        first_word = candidate.split()[0]
        if first_word in _COMMON_NON_PROPER:
            continue
        proper.append(candidate)

    years = [m.group(0) for m in _YEAR_RE.finditer(text or "")]
    year_set = set(years)
    numbers = [m.group(0) for m in _NUMBER_RE.finditer(text or "") if m.group(0) not in year_set]

    return {"proper_nouns": proper, "years": years, "numbers": numbers}


def _check_grounding(answer: Any, chunks: List[dict], mode: str) -> Dict[str, Any]:
    """
    Verify proper nouns and numbers in the answer appear in retrieved chunks.

    strict: any unverified proper noun or year fails grounding.
    truthful: tolerate up to 15% unverified proper nouns (rephrased project names etc.).
    off: always grounded.
    """
    if mode == "off" or not chunks:
        return {"grounded": True, "unverified": []}

    answer_text = _flatten_to_text(answer)
    if not answer_text.strip():
        return {"grounded": False, "unverified": []}

    haystack = " ".join(str(c.get("text") or "") for c in chunks).lower()

    tokens = _extract_verifiable_tokens(answer_text)
    unverified: List[str] = []

    for noun in tokens["proper_nouns"]:
        if noun.lower() not in haystack:
            unverified.append(noun)

    for year in tokens["years"]:
        if year not in haystack:
            unverified.append(year)

    if mode == "strict":
        grounded = len(unverified) == 0
    elif mode == "truthful":
        denom = max(len(tokens["proper_nouns"]) + len(tokens["years"]), 1)
        grounded = (len(unverified) / denom) < 0.15
    else:
        grounded = True

    # Deduplicate while preserving order
    seen: set = set()
    unverified_unique: List[str] = []
    for item in unverified:
        if item not in seen:
            seen.add(item)
            unverified_unique.append(item)

    return {"grounded": grounded, "unverified": unverified_unique}


# ---------------------------------------------------------------------------
# Core generation contract
# ---------------------------------------------------------------------------

def generate_with_output_contract(
    llm,
    prompt: str,
    gen: EffectiveGenerationConfig,
) -> Any:
    """
    Execute the generation contract:
      generate → parse → validate → retry with correction → explicit fail.
    """
    if gen.response_type != "json":
        t0 = time.monotonic()
        result = llm.generate(prompt, temperature=gen.temperature)
        logger.info("generate response_type=markdown latency_ms=%.1f", (time.monotonic() - t0) * 1000)
        return result

    if gen.strict and gen.schema is None:
        raise ValueError(
            "strict=True requires a schema. "
            "Define a schema in the generation config or task override."
        )

    schema = _normalize_schema(gen.schema)
    retry_prompt = prompt
    last_error: Optional[Exception] = None

    for attempt in range(gen.max_retries + 1):
        t0 = time.monotonic()
        output = llm.generate(retry_prompt, temperature=gen.temperature)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            parsed = _extract_json_payload(output)
            if schema:
                _validate_schema(parsed, schema)
            logger.info(
                "generate response_type=json attempt=%d/%d latency_ms=%.1f",
                attempt + 1,
                gen.max_retries + 1,
                latency_ms,
            )
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Structured generation attempt %d/%d failed: %s",
                attempt + 1,
                gen.max_retries + 1,
                exc,
            )
            if attempt < gen.max_retries:
                retry_prompt = (
                    f"{prompt}\n\n"
                    "CORRECTION: Your previous response was not valid JSON or did not "
                    "match the required schema. Return ONLY valid JSON. "
                    "Do not add prose, markdown fences, or comments."
                )

    if gen.strict:
        raise ValueError(
            f"Generation failed after {gen.max_retries + 1} attempt(s). "
            f"Last error: {last_error}"
        )

    logger.error(
        "generate FAILED (non-strict) after %d attempts: %s",
        gen.max_retries + 1,
        last_error,
    )
    return {"error": "generation_failed", "detail": str(last_error)}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _history_aware_retrieval_query(
    query: str,
    history: Optional[List[Dict[str, Any]]],
    summary: Optional[str],
) -> str:
    """Compose a retrieval query that includes the conversation tail.

    Helps resolve pronouns and follow-ups ("what about the next one?") at the
    retrieval stage even without an explicit query-rewrite LLM call (Phase 3).
    """
    if not history and not summary:
        return query
    parts: List[str] = []
    if summary and summary.strip():
        parts.append(summary.strip())
    if history:
        # Take the last assistant message (most recent context the user is following up on).
        for m in reversed(history):
            if m.get("role") == "assistant" and str(m.get("content") or "").strip():
                content = str(m["content"])
                parts.append(f"Recent context: {content[:500]}")
                break
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)


def generate_answer(
    ctx: ExecutionContext,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> dict:
    """
    Full RAG pipeline: retrieve → rerank → generate.
    All behavior is driven by ExecutionContext.

    Optional `chat_history` and `summary` enable multi-turn chat:
    - retrieval query is augmented with the recent context (helps resolve pronouns)
    - generation prompt includes a CONVERSATION HISTORY block (strict mode only)
    """
    gen = ctx.effective_generation
    mode = getattr(gen, "grounding_mode", "off") or "off"

    # If query rewriting is enabled the rewriter handles history-context via the
    # `summary` kwarg. Otherwise, fall back to baking history into the query string
    # so pronouns and follow-ups can resolve at retrieval time.
    if ctx.registry.retrieval.query_rewrite:
        retrieval_query = query
    else:
        retrieval_query = _history_aware_retrieval_query(query, chat_history, summary)

    t_retrieve = time.monotonic()
    retrieval_result = retrieve_and_rerank(ctx, retrieval_query, history_summary=summary)
    retrieve_ms = (time.monotonic() - t_retrieve) * 1000

    clarification = retrieval_result.get("clarification")

    if clarification:
        if gen.response_type == "json":
            context_str = clarification
        else:
            return {
                "answer": clarification,
                "grounded": False,
                "sources": [],
                "meta": {"retrieved_count": 0, "reranked_count": 0, "retrieve_ms": 0, "generate_ms": 0},
            }

    chunks = _dedupe_chunks(retrieval_result.get("chunks", []))
    context_str = _build_context_str(chunks, max_context_tokens=gen.max_context_tokens)

    if not context_str.strip():
        if mode == "strict":
            answer = STRICT_REFUSAL if gen.response_type != "json" else {
                "answer": STRICT_REFUSAL, "grounded": False, "citations": []
            }
            return {
                "answer": answer,
                "grounded": False,
                "sources": [],
                "meta": {"retrieved_count": 0, "reranked_count": 0, "retrieve_ms": round(retrieve_ms, 1), "generate_ms": 0},
            }
        if gen.response_type == "json":
            context_str = _NO_CONTEXT
        else:
            return {
                "answer": _NO_CONTEXT,
                "grounded": False,
                "sources": [],
                "meta": {"retrieved_count": 0, "reranked_count": 0, "retrieve_ms": round(retrieve_ms, 1), "generate_ms": 0},
            }

    prompt = _build_prompt(query, context_str, gen, history=chat_history, summary=summary)
    llm = get_llm(ctx.llm_config)

    t0 = time.monotonic()
    answer = generate_with_output_contract(llm, prompt, gen)
    generate_ms = (time.monotonic() - t0) * 1000

    cortex_logger.log_generate(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        response_type=gen.response_type,
        attempt_count=1,
        success=not (isinstance(answer, dict) and answer.get("error") == "generation_failed"),
        latency_ms=generate_ms,
    )

    # Post-generation grounding check
    grounding = _check_grounding(answer, chunks, mode)
    if mode == "strict" and not grounding["grounded"]:
        logger.warning(
            "Strict grounding check failed: unverified entities %s — replacing with refusal",
            grounding["unverified"][:5],
        )
        if gen.response_type == "json":
            answer = {"answer": STRICT_REFUSAL, "grounded": False, "citations": []}
        else:
            answer = STRICT_REFUSAL

    meta: Dict[str, Any] = {
        "retrieved_count": retrieval_result.get("retrieved_count", len(chunks)),
        "reranked_count": retrieval_result.get("reranked_count", len(chunks)),
        "retrieve_ms": round(retrieve_ms, 1),
        "generate_ms": round(generate_ms, 1),
    }
    if grounding["unverified"]:
        meta["unverified_entities"] = grounding["unverified"]

    return {
        "answer": answer,
        "grounded": grounding["grounded"],
        "sources": _build_public_sources(chunks),
        "meta": meta,
    }


def generate_direct(ctx: ExecutionContext, query: str, context: str = "") -> dict:
    """
    Generation without retrieval. Caller supplies the context explicitly.
    """
    gen = ctx.effective_generation
    mode = getattr(gen, "grounding_mode", "off") or "off"
    prompt = _build_prompt(query, context, gen)
    llm = get_llm(ctx.llm_config)

    t0 = time.monotonic()
    answer = generate_with_output_contract(llm, prompt, gen)
    total_ms = (time.monotonic() - t0) * 1000

    cortex_logger.log_generate(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        response_type=gen.response_type,
        attempt_count=1,
        success=not (isinstance(answer, dict) and answer.get("error") == "generation_failed"),
        latency_ms=total_ms,
    )

    # Grounding check against the caller-supplied context, treated as a single chunk.
    pseudo_chunks = [{"text": context}] if context else []
    grounding = _check_grounding(answer, pseudo_chunks, mode)
    if mode == "strict" and not grounding["grounded"]:
        if gen.response_type == "json":
            answer = {"answer": STRICT_REFUSAL, "grounded": False, "citations": []}
        else:
            answer = STRICT_REFUSAL

    result: Dict[str, Any] = {"answer": answer}
    if mode != "off":
        result["grounded"] = grounding["grounded"]
        if grounding["unverified"]:
            result["unverified_entities"] = grounding["unverified"]
    return result
