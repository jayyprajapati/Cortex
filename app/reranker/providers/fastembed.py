from __future__ import annotations

from typing import Any, Dict, List


from app.reranker.base import BaseReranker


class FastEmbedReranker(BaseReranker):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.model_name = options.get("model", "BAAI/bge-reranker-v2-m3")
        self._model = None

    def _get_model(self):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise ImportError("fastembed is not installed. Run: pip install fastembed")
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self.model_name)
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

        scores = list(self._get_model().rerank(query, [c["text"] for c in candidates]))

        if scores:
            lo, hi = min(scores), max(scores)
            span = hi - lo
            norm = [(s - lo) / span if span > 1e-9 else 1.0 for s in scores]
        else:
            norm = []

        for i, c in enumerate(candidates):
            c["rerank_score"] = norm[i]
            c["rerank_raw_score"] = scores[i]

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        diversity = self.options.get("diversity", 0.0)
        if diversity > 0:
            return self._apply_mmr(candidates, top_k, diversity)
        return candidates[:top_k]
