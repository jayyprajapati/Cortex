from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.registry.service import build_execution_context


@dataclass(frozen=True)
class PromptSpec:
    builder: Callable[..., str]
    structured_output: bool = False
    schema: Optional[Dict[str, Any]] = None


def _build_prompt(system_prompt: str, query: str, context: str, response_type: str) -> str:
    norm_sys = str(system_prompt or "").strip()
    safe_q = str(query or "").strip()
    safe_ctx = str(context or "").strip()

    if response_type == "json":
        return (
            f"{norm_sys}\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- Do not include markdown, code fences, or prose outside JSON.\n\n"
            f"Context:\n{safe_ctx}\n\n"
            f"Input:\n{safe_q}\n\n"
            "JSON Output:"
        )

    return f"{norm_sys}\n\nContext:\n{safe_ctx}\n\nQuestion:\n{safe_q}\n\nAnswer:"


def _normalize_schema(schema: Optional[Dict]) -> Optional[Dict]:
    if schema is None or "type" in schema:
        return schema

    type_map = {
        "string": "string", "number": "number", "array": "array",
        "object": "object", "boolean": "boolean", "integer": "integer", "null": "null",
    }
    properties: Dict[str, Any] = {}
    required = []
    for field_name, field_type in schema.items():
        t = type_map.get(str(field_type).strip().lower(), "string")
        properties[field_name] = {"type": t}
        required.append(field_name)
    return {"type": "object", "required": required, "properties": properties}


def get_prompt_spec(app_name: Optional[str], task: Optional[str] = None) -> PromptSpec:
    """
    Resolve a PromptSpec from the registry for the given app and task.
    Used by the /generate endpoint which doesn't go through ExecutionContext.
    """
    # Build a minimal context to resolve the effective generation config
    ctx = build_execution_context(
        app_name=app_name or "doclens",
        user_id="__prompt_spec_resolver__",
        task=task,
    )
    gen = ctx.effective_generation
    schema = _normalize_schema(gen.schema)

    def builder(query: str, context: str) -> str:
        return _build_prompt(gen.system_prompt, query, context, gen.response_type)

    return PromptSpec(
        builder=builder,
        structured_output=gen.response_type == "json",
        schema=schema,
    )


def get_prompt_builder(app_name: Optional[str], task: Optional[str] = None) -> Callable:
    return get_prompt_spec(app_name=app_name, task=task).builder


def is_structured_task(app_name: Optional[str], task: Optional[str] = None) -> bool:
    return get_prompt_spec(app_name=app_name, task=task).structured_output


def get_task_schema(app_name: Optional[str], task: Optional[str] = None) -> Optional[Dict]:
    return get_prompt_spec(app_name=app_name, task=task).schema
