from __future__ import annotations

from app.context import LLMConfig
from app.llm.base import BaseLLM
from app.llm.ollama_cloud import OllamaCloudLLM
from app.llm.ollama_local import OllamaLocalLLM
from app.llm.openai_llm import OpenAILLM


def get_llm(config: LLMConfig) -> BaseLLM:
    provider = config.provider

    if provider == "ollama_local":
        return OllamaLocalLLM(model=config.model)

    if provider == "ollama_cloud":
        if not config.api_key:
            raise ValueError("ollama_cloud provider requires api_key")
        return OllamaCloudLLM(api_key=config.api_key, model=config.model)

    if provider == "openai":
        if not config.api_key:
            raise ValueError("openai provider requires api_key")
        return OpenAILLM(api_key=config.api_key, model=config.model)

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. "
        "Valid providers: ollama_local, ollama_cloud, openai"
    )
