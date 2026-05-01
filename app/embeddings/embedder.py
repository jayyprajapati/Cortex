from __future__ import annotations

import logging
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.chunking.models import Chunk
from app.registry.models import EmbeddingConfig

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_embedding_dimension(model_name: str) -> int:
    model = _get_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError(f"Cannot determine embedding dimension for model: {model_name!r}")
    return int(dim)


def embed_chunks(chunks: List[Chunk], config: EmbeddingConfig) -> List:
    if not chunks:
        return []

    model = _get_model(config.model)
    texts = ["passage: " + c.text for c in chunks]
    t0 = time.monotonic()

    all_vectors: List = []
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i : i + config.batch_size]
        vectors = model.encode(batch, normalize_embeddings=config.normalize, show_progress_bar=False)
        all_vectors.extend(vectors)

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "embed_chunks model=%s count=%d latency_ms=%.1f",
        config.model,
        len(chunks),
        latency_ms,
    )

    return all_vectors


def embed_query(query: str, config: EmbeddingConfig) -> list:
    model = _get_model(config.model)
    t0 = time.monotonic()
    vector = model.encode(
        "query: " + query,
        normalize_embeddings=config.normalize,
        show_progress_bar=False,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("embed_query model=%s latency_ms=%.1f", config.model, latency_ms)
    return vector
