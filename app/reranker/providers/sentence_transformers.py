from __future__ import annotations

from typing import Any, Dict, List

from app.reranker.base import BaseReranker


class SentenceTransformersReranker(BaseReranker):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.model_name = options.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.diversity = options.get("diversity", 0.0)
        self._model = None

    def _get_model(self):
        from sentence_transformers import CrossEncoder
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        candidate_cap: int,
    ) -> List[Dict[str, Any]]:
        candidates = chunks[:candidate_cap]
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = self._get_model().predict(pairs).tolist()

        if raw_scores:
            lo, hi = min(raw_scores), max(raw_scores)
            span = hi - lo
            norm = [(s - lo) / span if span > 1e-9 else 1.0 for s in raw_scores]
        else:
            norm = []

        for i, c in enumerate(candidates):
            c["rerank_score"] = norm[i]
            c["rerank_raw_score"] = raw_scores[i]

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        if self.diversity > 0:
            return self._apply_mmr(candidates, top_k, self.diversity)
        return candidates[:top_k]
