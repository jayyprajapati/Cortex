from app.config import (
    ALLOW_DEFAULT_LLM,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_CLOUD_API_KEY,
    OPENAI_API_KEY,
)
from app.pipeline.retrieve_pipeline import retrieve
from app.llm.factory import get_llm
from cortex.prompts.registry import get_prompt_builder


NO_CONTEXT_FALLBACK = "No relevant information found."


def _resolve_requested_llm_config(llm_config):
    provider = str(llm_config.get("provider") or "").strip().lower()
    model = llm_config.get("model")
    api_key = str(llm_config.get("api_key") or "").strip()

    if provider == "openai":
        resolved_api_key = api_key

        if not resolved_api_key and ALLOW_DEFAULT_LLM:
            resolved_api_key = OPENAI_API_KEY

        if not resolved_api_key:
            raise ValueError("An API key is required for provider=openai.")

        config = {
            "provider": "openai",
            "api_key": resolved_api_key,
        }

    elif provider in {"ollama", "ollama_cloud", "ollama_local"}:
        if provider == "ollama_local":
            config = {"provider": "ollama_local"}
        elif api_key:
            config = {
                "provider": "ollama_cloud",
                "api_key": api_key,
            }
        elif ALLOW_DEFAULT_LLM and OLLAMA_CLOUD_API_KEY:
            config = {
                "provider": "ollama_cloud",
                "api_key": OLLAMA_CLOUD_API_KEY,
            }
        else:
            config = {"provider": "ollama_local"}
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if model:
        config["model"] = model

    return config


def resolve_llm_config(llm_config=None):
    if llm_config:
        return _resolve_requested_llm_config(llm_config)

    provider = (LLM_PROVIDER or "ollama_cloud").strip().lower()
    model = LLM_MODEL

    if provider in {"ollama", "ollama_cloud"} and OLLAMA_CLOUD_API_KEY:
        config = {
            "provider": "ollama_cloud",
            "api_key": OLLAMA_CLOUD_API_KEY,
        }
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        config = {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
        }
    elif provider in {"ollama", "ollama_local", "ollama_cloud"}:
        config = {"provider": "ollama_local"}
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if model:
        config["model"] = model

    return config


def build_context(chunks):
    context_blocks = []

    for i, chunk in enumerate(chunks):
        section = str(chunk.get("section") or "Untitled").strip()
        page = chunk.get("page")
        page_label = page if page is not None else "N/A"

        context_blocks.append(
            f"[Source {i+1}. Section: {section}. Page: {page_label}]\n{chunk['text']}"
        )

    return "\n\n".join(context_blocks)


def build_prompt(query, chunks, app_name="default"):
    context = build_context(chunks)
    prompt_builder = get_prompt_builder(app_name)
    return prompt_builder(query=query, context=context)


def dedupe_chunks(chunks):
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        key = (
            str(chunk.get("doc_id") or "").strip(),
            chunk.get("page"),
            str(chunk.get("section") or "").strip(),
            " ".join(str(chunk.get("text") or "").split()),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    return unique_chunks


def _build_doc_sources(doc_ids):
    return [{"doc_id": doc_id} for doc_id in doc_ids if doc_id]


def generate_answer(query, *args, user_id=None, app_name="default", doc_id=None, llm_config=None):
    resolved_user_id = user_id
    resolved_app_name = app_name
    resolved_doc_id = doc_id
    resolved_llm_config = llm_config

    if args:
        first_arg = args[0]

        # Backward-compatible form: generate_answer(query, llm_config, user_id, doc_id)
        if isinstance(first_arg, dict):
            resolved_llm_config = first_arg

            if len(args) > 1 and resolved_user_id is None:
                resolved_user_id = args[1]

            if len(args) > 2 and resolved_doc_id is None:
                resolved_doc_id = args[2]
        else:
            # New positional form: generate_answer(query, user_id, app_name, doc_id)
            if resolved_user_id is None:
                resolved_user_id = first_arg

            if len(args) > 1:
                resolved_app_name = args[1]

            if len(args) > 2 and resolved_doc_id is None:
                resolved_doc_id = args[2]

            if len(args) > 3 and resolved_llm_config is None and isinstance(args[3], dict):
                resolved_llm_config = args[3]

    if not resolved_user_id:
        raise ValueError("user_id is required")

    retrieval_result = retrieve(
        query,
        user_id=resolved_user_id,
        doc_id=resolved_doc_id,
        return_meta=True,
    )

    if isinstance(retrieval_result, dict):
        chunks = retrieval_result.get("chunks", [])
        clarification = retrieval_result.get("clarification")
        doc_ids = retrieval_result.get("doc_ids", [])
    else:
        chunks = retrieval_result
        clarification = None
        doc_ids = []

    if clarification:
        return {
            "answer": clarification,
            "sources": _build_doc_sources(doc_ids),
        }

    chunks = dedupe_chunks(chunks)

    context = build_context(chunks)
    if not context.strip():
        return {
            "answer": NO_CONTEXT_FALLBACK,
            "sources": [],
        }

    prompt_builder = get_prompt_builder(resolved_app_name)
    prompt = prompt_builder(query=query, context=context)

    llm = get_llm(resolve_llm_config(resolved_llm_config))
    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "sources": chunks,
    }