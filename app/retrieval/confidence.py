from __future__ import annotations

from typing import Dict, List, Literal

ConfidenceLevel = Literal["high", "medium", "low"]


def compute_confidence(
    chunks: List[Dict],
    min_score: float = 0.25,
) -> ConfidenceLevel:
    """
    Derive a confidence level from rerank or retrieval scores.
    high: top score > 0.7
    medium: top score > min_score
    low: no scores or top score <= min_score
    """
    if not chunks:
        return "low"

    scores = [
        c.get("rerank_score") or c.get("score", 0.0)
        for c in chunks
    ]
    scores = [s for s in scores if s is not None]
    if not scores:
        return "low"

    top = max(scores)
    if top > 0.7:
        return "high"
    if top > min_score:
        return "medium"
    return "low"
