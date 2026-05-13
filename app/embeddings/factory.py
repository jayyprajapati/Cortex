from __future__ import annotations

from typing import Any, Dict

from app.embeddings.base import BaseEmbedder


def get_embedder(provider: str | None, options: Dict[str, Any]) -> BaseEmbedder:
    if not provider:
        provider = "sentence_transformers"

    if provider == "fastembed":
        from app.embeddings.providers.fastembed import FastEmbedEmbedder
        return FastEmbedEmbedder(options)
    elif provider == "sentence_transformers":
        from app.embeddings.providers.sentence_transformers import SentenceTransformersEmbedder
        return SentenceTransformersEmbedder(options)
    elif provider == "openai":
        from app.embeddings.providers.openai import OpenAIEmbedder
        return OpenAIEmbedder(options)
    elif provider == "cohere":
        from app.embeddings.providers.cohere import CohereEmbedder
        return CohereEmbedder(options)
    else:
        raise ValueError(f"Unknown embedder provider: {provider!r}")
