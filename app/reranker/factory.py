from __future__ import annotations

from typing import Any, Dict

from app.reranker.base import BaseReranker


def get_reranker(provider: str | None, options: Dict[str, Any]) -> BaseReranker:
    if not provider:
        provider = "sentence_transformers"

    if provider == "fastembed":
        from app.reranker.providers.fastembed import FastEmbedReranker
        return FastEmbedReranker(options)

    if provider == "sentence_transformers":
        from app.reranker.providers.sentence_transformers import SentenceTransformersReranker
        return SentenceTransformersReranker(options)

    if provider == "cohere":
        from app.reranker.providers.cohere import CohereReranker
        return CohereReranker(options)

    if provider == "jina":
        from app.reranker.providers.jina import JinaReranker
        return JinaReranker(options)

    raise ValueError(f"Unknown reranker provider: {provider!r}")
