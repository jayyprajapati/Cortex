from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.context import EffectiveGenerationConfig, ExecutionContext, LLMConfig
from app.registry.models import ApplicationConfig, TaskOverride
from app.registry.store import get_app


def _resolve_llm_config(llm_override: Optional[Dict[str, Any]] = None) -> LLMConfig:
    from app.config import LLM_MODEL, LLM_PROVIDER, OLLAMA_CLOUD_API_KEY, OPENAI_API_KEY

    if llm_override:
        provider = str(llm_override.get("provider") or "").strip().lower()
        model = llm_override.get("model")
        api_key = str(llm_override.get("api_key") or "").strip()

        if provider == "openai":
            resolved_key = api_key or OPENAI_API_KEY
            if not resolved_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI provider requires OPENAI_API_KEY or api_key override",
                )
            return LLMConfig(provider="openai", model=model or "gpt-4o-mini", api_key=resolved_key)

        if provider in {"ollama_local", "ollama"}:
            base_url = str(llm_override.get("base_url") or "").strip() or None
            return LLMConfig(provider="ollama_local", model=model or "llama3", base_url=base_url)

        if provider == "ollama_cloud":
            resolved_key = api_key or OLLAMA_CLOUD_API_KEY
            if not resolved_key:
                return LLMConfig(provider="ollama_local", model=model or "llama3")
            return LLMConfig(provider="ollama_cloud", model=model or "gpt-oss:120b", api_key=resolved_key)

        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider!r}")

    env_provider = (LLM_PROVIDER or "ollama_cloud").strip().lower()
    env_model = LLM_MODEL

    if env_provider in {"ollama_cloud", "ollama"} and OLLAMA_CLOUD_API_KEY:
        return LLMConfig(
            provider="ollama_cloud",
            model=env_model or "gpt-oss:120b",
            api_key=OLLAMA_CLOUD_API_KEY,
        )

    if env_provider == "openai" and OPENAI_API_KEY:
        return LLMConfig(
            provider="openai",
            model=env_model or "gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

    return LLMConfig(provider="ollama_local", model=env_model or "llama3")


def _resolve_effective_generation(
    config: ApplicationConfig,
    task: Optional[str],
    prompt_override: Optional[str],
) -> EffectiveGenerationConfig:
    base = config.generation
    system_prompt = config.defaults.system_prompt

    resolved_task = task or config.default_task
    task_override: Optional[TaskOverride] = None
    if resolved_task and resolved_task in config.tasks:
        task_override = config.tasks[resolved_task]

    if task_override:
        if task_override.system_prompt:
            system_prompt = task_override.system_prompt
        response_type = task_override.response_type or base.response_type
        schema = task_override.output_schema if task_override.output_schema is not None else base.output_schema
        temperature = task_override.temperature if task_override.temperature is not None else base.temperature
        strict = task_override.strict if task_override.strict is not None else base.strict
        max_retries = task_override.max_retries if task_override.max_retries is not None else base.max_retries
    else:
        response_type = base.response_type
        schema = base.output_schema
        temperature = base.temperature
        strict = base.strict
        max_retries = base.max_retries

    if prompt_override:
        system_prompt = prompt_override.strip()

    return EffectiveGenerationConfig(
        system_prompt=system_prompt,
        response_type=response_type,
        schema=schema,
        temperature=temperature,
        strict=strict,
        max_retries=max_retries,
    )


def build_execution_context(
    app_name: str,
    user_id: str,
    task: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    llm_override: Optional[Dict[str, Any]] = None,
    prompt_override: Optional[str] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    normalized_app = (app_name or "").strip().lower()
    if not normalized_app:
        raise HTTPException(status_code=400, detail="app_name is required")

    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")

    config = get_app(normalized_app)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown application: {normalized_app!r}")

    normalized_task = (task or "").strip().lower() or None
    if normalized_task and normalized_task not in config.tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task {normalized_task!r} for application {normalized_app!r}",
        )

    llm_config = _resolve_llm_config(llm_override)
    effective_gen = _resolve_effective_generation(config, normalized_task, prompt_override)

    clean_doc_ids: Optional[List[str]] = None
    if doc_ids:
        clean_doc_ids = [str(d).strip() for d in doc_ids if str(d).strip()] or None

    return ExecutionContext(
        app_name=normalized_app,
        user_id=normalized_user,
        registry=config,
        llm_config=llm_config,
        effective_generation=effective_gen,
        doc_ids=clean_doc_ids,
        task=normalized_task,
        prompt_override=prompt_override,
        request_overrides=request_overrides or {},
    )
