from __future__ import annotations

from typing import Any, Dict

from app.vectorstore.base import BaseVectorStore


def get_vector_store(provider: str | None, options: Dict[str, Any]) -> BaseVectorStore:
    if not provider:
        provider = "qdrant"

    if provider == "qdrant":
        from app.vectorstore.providers.qdrant import QdrantVectorStore
        return QdrantVectorStore(options)

    if provider == "pinecone":
        from app.vectorstore.providers.pinecone import PineconeVectorStore
        return PineconeVectorStore(options)

    if provider == "weaviate":
        from app.vectorstore.providers.weaviate import WeaviateVectorStore
        return WeaviateVectorStore(options)

    if provider == "chroma":
        from app.vectorstore.providers.chroma import ChromaVectorStore
        return ChromaVectorStore(options)

    raise ValueError(f"Unknown vector store provider: {provider!r}")
