from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.context import EffectiveGenerationConfig, ExecutionContext
from app.llm.factory import get_llm
from app.observability.logger import cortex_logger

logger = logging.getLogger(__name__)


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
# Context and chunk utilities
# ---------------------------------------------------------------------------

def _build_context_str(chunks: List[dict], max_context_tokens: int = 4000) -> str:
    # Drop lowest-ranked chunks (end of list) until the context fits in the budget.
    # Use tiktoken for accurate token counting.
    from app.chunking.tokenizer import token_count
    budget = max_context_tokens
    kept: List[dict] = []
    for chunk in chunks:
        cost = token_count(str(chunk.get("text") or ""))
        if cost > budget and kept:
            break
        kept.append(chunk)
        budget -= cost

    blocks = []
    for i, chunk in enumerate(kept):
        section_path = chunk.get("section_path") or chunk.get("hierarchy") or []
        if section_path and isinstance(section_path, list):
            section_label = " > ".join(str(s) for s in section_path if s != "_root")
        else:
            section_label = str(chunk.get("section") or "Untitled").strip()
        page = chunk.get("page")
        page_label = page if page is not None else "N/A"
        blocks.append(f"[Source {i+1}. Section: {section_label}. Page: {page_label}]\n{chunk['text']}")
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


def _validate_citations(answer_text: str, num_sources: int, threshold: float = 0.7) -> dict:
    """Validate [N] citation markers in markdown answer.

    Returns: {valid: bool, coverage: float, out_of_range: list[int]}
    - coverage: fraction of sentences with at least one [N] citation
    - out_of_range: [N] values that exceed num_sources
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    if not sentences:
        return {"valid": True, "coverage": 1.0, "out_of_range": []}

    cited_count = sum(1 for s in sentences if _CITATION_RE.search(s))
    coverage = cited_count / len(sentences)

    out_of_range = []
    for match in _CITATION_RE.finditer(answer_text):
        for piece in match.group(1).split(","):
            try:
                idx = int(piece.strip())
                if idx < 1 or idx > num_sources:
                    out_of_range.append(idx)
            except ValueError:
                pass
    out_of_range = sorted(set(out_of_range))

    valid = coverage >= threshold and not out_of_range
    return {"valid": valid, "coverage": round(coverage, 3), "out_of_range": out_of_range}


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


def _check_grounding(answer: Any, chunks: List[dict], mode: str, unverified_threshold: float = 0.15) -> Dict[str, Any]:
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
        grounded = (len(unverified) / denom) < unverified_threshold
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
# Prompt rendering (Jinja2 templates)
# ---------------------------------------------------------------------------

def _render_prompt(
    query: str,
    context: str,
    gen: EffectiveGenerationConfig,
    history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> str:
    from app.prompts.loader import render
    mode = getattr(gen, "grounding_mode", "off") or "off"

    if gen.response_type == "json":
        return render(
            "json_task.j2",
            system_prompt=gen.system_prompt.strip(),
            context=context,
            query=query,
            history=history or [],
            summary=summary or "",
        )

    refusal_instruction = (
        "Tell the user you don't have enough information to answer this question "
        "in your own words. Vary the phrasing — do not use a fixed phrase."
    )

    voice_footer = (getattr(gen, "voice_footer", None) or "").strip()

    return render(
        "chat.md.j2",
        system_prompt=gen.system_prompt.strip(),
        context=context,
        query=query,
        history=history or [],
        summary=summary or "",
        grounding_mode=mode,
        refusal_instruction=refusal_instruction,
        voice_footer=voice_footer,
    )


def _render_clarification_prompt(
    query: str,
    clarifying_question: str,
    gen: EffectiveGenerationConfig,
) -> str:
    from app.prompts.loader import render
    return render(
        "clarify.md.j2",
        system_prompt=gen.system_prompt.strip(),
        query=query,
        clarifying_question=clarifying_question,
    )


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
        raise ValueError("strict=True requires a schema.")

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
            logger.info("generate json attempt=%d/%d latency_ms=%.1f", attempt+1, gen.max_retries+1, latency_ms)
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning("JSON generation attempt %d/%d failed: %s", attempt+1, gen.max_retries+1, exc)
            if attempt < gen.max_retries:
                from app.prompts.loader import render
                schema_hint = json.dumps(schema, indent=2) if schema else ""
                retry_prompt = render(
                    "grounding_correction.j2",
                    original_prompt=prompt,
                    error=str(exc),
                    schema_hint=schema_hint,
                )

    if gen.strict:
        raise ValueError(f"Generation failed after {gen.max_retries+1} attempt(s). Last error: {last_error}")
    logger.error("generate FAILED (non-strict) after %d attempts: %s", gen.max_retries+1, last_error)
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


async def stream_answer(
    ctx: ExecutionContext,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
    clarification_reply: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline with SSE streaming.
    Yields SSE event strings: meta → (clarification)? → delta* → citations → done | error
    """
    from app.streaming.sse import (
        meta_event, delta_event, clarification_event,
        citations_event, done_event, error_event,
    )
    from app.retrieval.pipeline import run_retrieval

    gen = ctx.effective_generation

    # Resolve last assistant message for query analysis context
    last_assistant = None
    if chat_history:
        for m in reversed(chat_history):
            if m.get("role") == "assistant":
                last_assistant = str(m.get("content") or "")[:500]
                break

    if ctx.registry.retrieval.query_rewrite:
        retrieval_query = query
    else:
        retrieval_query = _history_aware_retrieval_query(query, chat_history, summary)

    import time as _time
    t_retrieve = _time.monotonic()

    try:
        retrieval_result = run_retrieval(
            ctx=ctx,
            query=retrieval_query,
            history_summary=summary,
            last_assistant_msg=last_assistant,
            clarification_reply=clarification_reply,
        )
    except Exception as exc:
        yield error_event(str(exc), code="retrieval_error")
        return

    retrieve_ms = (_time.monotonic() - t_retrieve) * 1000

    # Emit meta
    yield meta_event({
        "retrieved_count": retrieval_result.retrieved_count,
        "reranked_count": retrieval_result.reranked_count,
        "confidence": retrieval_result.confidence,
        "retrieve_ms": round(retrieve_ms, 1),
    })

    # Handle clarification
    if retrieval_result.needs_clarification:
        clarifying_q = retrieval_result.clarifying_question or "Could you provide more context?"
        clarification_text = _render_clarification_prompt(query, clarifying_q, gen)
        llm = get_llm(ctx.llm_config)
        clarification_answer = llm.generate(clarification_text, temperature=0.7)
        yield clarification_event(str(clarification_answer or "").strip())
        yield done_event({"type": "clarification"})
        return

    # Build context
    chunks = _dedupe_chunks(retrieval_result.chunks)
    context_str = _build_context_str(chunks, max_context_tokens=gen.max_context_tokens)

    if not context_str.strip():
        # No relevant context — let LLM respond naturally (no hardcoded refusal)
        no_context_prompt = _render_prompt(query, "", gen, history=chat_history, summary=summary)
        llm = get_llm(ctx.llm_config)
        answer = llm.generate(no_context_prompt, temperature=gen.temperature)
        answer_text = str(answer or "").strip()
        yield delta_event(answer_text)
        yield citations_event([])
        yield done_event({"grounded": False})
        return

    prompt = _render_prompt(query, context_str, gen, history=chat_history, summary=summary)
    llm = get_llm(ctx.llm_config)

    t0 = _time.monotonic()
    try:
        answer = generate_with_output_contract(llm, prompt, gen)
    except Exception as exc:
        yield error_event(str(exc), code="generation_error")
        return
    generate_ms = (_time.monotonic() - t0) * 1000

    answer_text = answer if isinstance(answer, str) else json.dumps(answer)
    yield delta_event(answer_text)

    sources = _build_public_sources(chunks)
    citations = extract_citations(answer_text, sources)
    yield citations_event(citations)

    mode = getattr(gen, "grounding_mode", "off") or "off"
    grounding = _check_grounding(answer, chunks, mode)

    yield done_event({
        "grounded": grounding["grounded"],
        "generate_ms": round(generate_ms, 1),
        "unverified_entities": grounding.get("unverified", []),
    })


def generate_answer(
    ctx: ExecutionContext,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
    clarification_reply: bool = False,
) -> dict:
    """Full RAG pipeline — non-streaming. Calls the new retrieval pipeline."""
    from app.retrieval.pipeline import run_retrieval
    gen = ctx.effective_generation
    mode = getattr(gen, "grounding_mode", "off") or "off"

    last_assistant = None
    if chat_history:
        for m in reversed(chat_history):
            if m.get("role") == "assistant":
                last_assistant = str(m.get("content") or "")[:500]
                break

    if ctx.registry.retrieval.query_rewrite:
        retrieval_query = query
    else:
        retrieval_query = _history_aware_retrieval_query(query, chat_history, summary)

    t_retrieve = time.monotonic()
    retrieval_result = run_retrieval(
        ctx=ctx,
        query=retrieval_query,
        history_summary=summary,
        last_assistant_msg=last_assistant,
        clarification_reply=clarification_reply,
    )
    retrieve_ms = (time.monotonic() - t_retrieve) * 1000

    # Handle clarification
    if retrieval_result.needs_clarification:
        clarifying_q = retrieval_result.clarifying_question or "Could you provide more context?"
        clarification_text = _render_clarification_prompt(query, clarifying_q, gen)
        llm = get_llm(ctx.llm_config)
        clarification_answer = llm.generate(clarification_text, temperature=0.7)
        return {
            "answer": str(clarification_answer or "").strip(),
            "grounded": False,
            "sources": [],
            "needs_clarification": True,
            "meta": {
                "retrieved_count": 0,
                "reranked_count": 0,
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": 0,
            },
        }

    chunks = _dedupe_chunks(retrieval_result.chunks)
    context_str = _build_context_str(chunks, max_context_tokens=gen.max_context_tokens)

    if not context_str.strip():
        return {
            "answer": "I don't have enough information in my knowledge base to answer this question.",
            "grounded": False,
            "sources": [],
            "confidence": "low",
            "meta": {
                "retrieved_count": retrieval_result.retrieved_count,
                "reranked_count": retrieval_result.reranked_count,
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": 0,
            },
        }

    prompt = _render_prompt(query, context_str, gen, history=chat_history, summary=summary)
    llm = get_llm(ctx.llm_config)
    t0 = time.monotonic()
    answer = generate_with_output_contract(llm, prompt, gen)
    generate_ms = (time.monotonic() - t0) * 1000

    # P1.1: Citation validation (markdown only, when enabled)
    citation_confidence_low = False
    if gen.response_type == "markdown" and getattr(gen, "citation_validation", False):
        citation_result = _validate_citations(
            str(answer), len(chunks), threshold=getattr(gen, "citation_threshold", 0.7)
        )
        if not citation_result["valid"]:
            correction = (
                f"Your previous answer had citation issues "
                f"(coverage: {citation_result['coverage']:.0%}, "
                f"out-of-range: {citation_result['out_of_range']}). "
                f"Please rewrite with [N] citations for every factual claim, "
                f"where N is 1-{len(chunks)}.\n\n"
                f"Original question: {query}\n\nAnswer:"
            )
            t_corr = time.monotonic()
            corrected = llm.generate(correction, temperature=gen.temperature)
            generate_ms += (time.monotonic() - t_corr) * 1000
            corrected_text = str(corrected or "").strip()
            if corrected_text:
                answer = corrected_text
            citation_result2 = _validate_citations(
                str(answer), len(chunks), threshold=getattr(gen, "citation_threshold", 0.7)
            )
            if not citation_result2["valid"]:
                logger.warning(
                    "Citation validation still failing after correction: coverage=%.2f out_of_range=%s",
                    citation_result2["coverage"], citation_result2["out_of_range"],
                )
                citation_confidence_low = True

    cortex_logger.log_generate(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        response_type=gen.response_type,
        attempt_count=1,
        success=not (isinstance(answer, dict) and answer.get("error") == "generation_failed"),
        latency_ms=generate_ms,
    )

    _unverified_thresh = getattr(gen, "grounding_unverified_threshold", 0.15)
    grounding = _check_grounding(answer, chunks, mode, unverified_threshold=_unverified_thresh)
    if mode == "strict" and not grounding["grounded"]:
        logger.warning("Strict grounding failed: unverified=%s", grounding["unverified"][:5])
        correction = (
            f"Your previous answer contained unverified claims: {grounding['unverified'][:5]}. "
            f"Rewrite the answer removing or correcting any claim not supported by the provided sources. "
            f"If you cannot answer without those claims, say 'I don't have enough information.'\n\n"
            f"Original question: {query}\n\nAnswer:"
        )
        t_corr = time.monotonic()
        corrected = llm.generate(correction, temperature=gen.temperature)
        generate_ms += (time.monotonic() - t_corr) * 1000
        answer = str(corrected or "").strip() or answer
        grounding = _check_grounding(answer, chunks, mode)
        if not grounding["grounded"]:
            answer = "I don't have enough information in the provided sources to answer this question accurately."
            grounding = {"grounded": False, "unverified": grounding["unverified"]}

    meta: Dict[str, Any] = {
        "retrieved_count": retrieval_result.retrieved_count,
        "reranked_count": retrieval_result.reranked_count,
        "retrieve_ms": round(retrieve_ms, 1),
        "generate_ms": round(generate_ms, 1),
    }
    if grounding["unverified"]:
        meta["unverified_entities"] = grounding["unverified"]
    if citation_confidence_low:
        meta["confidence"] = "low"

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
    prompt = _render_prompt(query, context, gen)
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
        logger.warning("Strict grounding failed in generate_direct: unverified=%s", grounding["unverified"][:5])
        # Don't replace with hardcoded string — grounded=False signals this to caller

    result: Dict[str, Any] = {"answer": answer}
    if mode != "off":
        result["grounded"] = grounding["grounded"]
        if grounding["unverified"]:
            result["unverified_entities"] = grounding["unverified"]
    return result
