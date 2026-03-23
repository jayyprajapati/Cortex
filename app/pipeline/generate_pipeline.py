import os

from app.config import OLLAMA_CLOUD_API_KEY
from app.pipeline.retrieve_pipeline import retrieve
from app.llm.factory import get_llm
from cortex.prompts.registry import get_prompt_builder


NO_CONTEXT_FALLBACK = "I could not find enough information in your documents to answer that question."


def resolve_llm_config(llm_config=None):
    if llm_config:
        return llm_config

    provider = os.getenv("LLM_PROVIDER", "ollama_cloud")
    model = os.getenv("LLM_MODEL")

    if provider == "ollama_cloud":
        config = {
            "provider": "ollama_cloud",
            "api_key": OLLAMA_CLOUD_API_KEY,
        }
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        config = {
            "provider": "openai",
            "api_key": api_key,
        }
    else:
        config = {"provider": "ollama_local"}

    if model:
        config["model"] = model

    return config


def build_context(chunks):
    context_blocks = []

    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1} | Section: {chunk.get('section')} | Page: {chunk.get('page')}]\n{chunk['text']}"
        )

    return "\n\n".join(context_blocks)


def build_prompt(query, chunks, app_name="default"):
    context = build_context(chunks)
    prompt_builder = get_prompt_builder(app_name)
    return prompt_builder(query=query, context=context)


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

    chunks = retrieve(query, user_id=resolved_user_id, doc_id=resolved_doc_id)

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