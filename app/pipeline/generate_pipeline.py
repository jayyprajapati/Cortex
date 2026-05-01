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

def _build_prompt(query: str, context: str, gen: EffectiveGenerationConfig) -> str:
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


def _build_context_str(chunks: List[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks):
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

def generate_answer(ctx: ExecutionContext, query: str) -> dict:
    """
    Full RAG pipeline: retrieve → rerank → generate.
    All behavior is driven by ExecutionContext.
    """
    gen = ctx.effective_generation

    retrieval_result = retrieve_and_rerank(ctx, query)
    clarification = retrieval_result.get("clarification")

    if clarification:
        if gen.response_type == "json":
            context_str = clarification
        else:
            return {"answer": clarification, "sources": []}

    chunks = _dedupe_chunks(retrieval_result.get("chunks", []))
    context_str = _build_context_str(chunks)

    if not context_str.strip():
        if gen.response_type == "json":
            context_str = _NO_CONTEXT
        else:
            return {"answer": _NO_CONTEXT, "sources": []}

    prompt = _build_prompt(query, context_str, gen)
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

    return {
        "answer": answer,
        "sources": _build_public_sources(chunks),
    }


def generate_direct(ctx: ExecutionContext, query: str, context: str = "") -> dict:
    """
    Generation without retrieval. Caller supplies the context explicitly.
    """
    gen = ctx.effective_generation
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

    return {"answer": answer}
