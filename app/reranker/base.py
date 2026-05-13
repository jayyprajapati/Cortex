from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        candidate_cap: int,
    ) -> List[Dict[str, Any]]:
        ...

    def _apply_mmr(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int,
        diversity: float,
    ) -> List[Dict[str, Any]]:
        if not chunks or diversity <= 0.0 or len(chunks) <= 1:
            return chunks[:top_k]
        lam = 1.0 - max(0.0, min(1.0, diversity))
        token_sets = [_token_set(c.get("text") or "") for c in chunks]
        remaining = list(range(len(chunks)))
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
