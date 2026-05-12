from __future__ import annotations

import logging
import re
import time
from typing import List

from sentence_transformers import CrossEncoder

from app.registry.models import RerankingConfig

logger = logging.getLogger(__name__)

_model_cache: dict[str, CrossEncoder] = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def _token_set(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _apply_mmr(
    chunks: List[dict],
    top_k: int,
    diversity: float,
) -> List[dict]:
    """Maximal Marginal Relevance — balances rerank_score with content diversity.

    diversity=0 → pure relevance (no change vs. raw rerank order).
    diversity=1 → pure diversity (always pick the most-different next chunk).

    Similarity is Jaccard over token sets — cheap, deterministic, no extra embedding pass.
    """
    if not chunks or diversity <= 0.0 or len(chunks) <= 1:
        return chunks[:top_k]

    lam = 1.0 - max(0.0, min(1.0, diversity))
    token_sets = [_token_set(c.get("text") or "") for c in chunks]
    remaining = list(range(len(chunks)))

    # Seed with the highest-rerank-score chunk
    remaining.sort(key=lambda i: chunks[i].get("rerank_score", 0.0), reverse=True)
    selected: List[int] = [remaining.pop(0)]

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_idx = 0
        for pos, i in enumerate(remaining):
            relevance = chunks[i].get("rerank_score", 0.0)
            max_sim = max(_jaccard(token_sets[i], token_sets[s]) for s in selected)
            mmr_score = lam * relevance - (1.0 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = pos
        selected.append(remaining.pop(best_idx))

    return [chunks[i] for i in selected]


def rerank(query: str, chunks: List[dict], config: RerankingConfig) -> List[dict]:
    """
    Rerank chunks using the configured cross-encoder.

    - Caps candidates to config.candidate_cap before scoring.
    - Attaches normalized rerank_score (0-1) and raw rerank_raw_score to each chunk.
    - When `config.diversity > 0`, applies MMR over the scored candidates so the
      final top_k balances relevance with content diversity.
    - Returns top config.top_k chunks.
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

    diversity = getattr(config, "diversity", 0.0) or 0.0
    if diversity > 0.0:
        result = _apply_mmr(candidates, top_k=config.top_k, diversity=diversity)
    else:
        result = candidates[: config.top_k]

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "rerank model=%s candidates=%d selected=%d diversity=%.2f latency_ms=%.1f",
        config.model,
        len(candidates),
        len(result),
        diversity,
        latency_ms,
    )

    return result
