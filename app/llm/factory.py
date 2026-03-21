from app.llm.ollama_local import OllamaLocalLLM
from app.llm.ollama_cloud import OllamaCloudLLM
from app.llm.openai_llm import OpenAILLM


def get_llm(config):
    provider = config["provider"]

    if provider == "ollama_local":
        return OllamaLocalLLM(model=config.get("model", "llama3"))

    elif provider == "ollama_cloud":
        return OllamaCloudLLM(
            api_key=config["api_key"],
            model=config.get("model", "gpt-oss:120b")
        )

    elif provider == "openai":
        return OpenAILLM(
            api_key=config["api_key"],
            model=config.get("model", "gpt-4o-mini")
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")