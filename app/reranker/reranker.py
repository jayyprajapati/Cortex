from __future__ import annotations

import logging
import time
from typing import List

from sentence_transformers import CrossEncoder

from app.registry.models import RerankingConfig

logger = logging.getLogger(__name__)

_model_cache: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _model_cache:
        logger.info("Loading reranker model: %s", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def rerank(query: str, chunks: List[dict], config: RerankingConfig) -> List[dict]:
    """
    Rerank chunks using the configured cross-encoder.

    - Caps candidates to config.candidate_cap before scoring.
    - Attaches normalized rerank_score (0-1) and raw rerank_raw_score to each chunk.
    - Returns top config.top_k chunks sorted by rerank_score descending.
    - Returns chunks unchanged (up to top_k) when reranking is disabled.
    """
    if not config.enabled:
        return chunks[: config.top_k]

    if not chunks:
        return chunks

    t0 = time.monotonic()
    candidates = chunks[: config.candidate_cap]
    model = _get_model(config.model)

    pairs = [(query, c["text"]) for c in candidates]
    raw_scores: List[float] = model.predict(pairs).tolist()
    norm_scores = _normalize(raw_scores)

    for chunk, norm, raw in zip(candidates, norm_scores, raw_scores):
        chunk["rerank_score"] = norm
        chunk["rerank_raw_score"] = raw

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    result = candidates[: config.top_k]

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "rerank model=%s candidates=%d selected=%d latency_ms=%.1f",
        config.model,
        len(candidates),
        len(result),
        latency_ms,
    )

    return result
