from __future__ import annotations

from typing import Any, Dict, List

from app.reranker.base import BaseReranker


class CohereReranker(BaseReranker):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.model_name = options.get("model", "rerank-english-v3.0")
        self.api_key = options.get("api_key")
        self.diversity = options.get("diversity", 0.0)
        self._client = None

    def _get_client(self):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is not installed. Run: pip install cohere")
        if self._client is None:
            self._client = cohere.Client(api_key=self.api_key)
        return self._client

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

        response = self._get_client().rerank(
            query=query,
            documents=[c["text"] for c in candidates],
            model=self.model_name,
            top_n=top_k,
        )

        scores_by_index: Dict[int, float] = {
            r.index: r.relevance_score for r in response.results
        }

        for i, c in enumerate(candidates):
            score = scores_by_index.get(i, 0.0)
            c["rerank_raw_score"] = score
            c["rerank_score"] = score

        scored = [c for i, c in enumerate(candidates) if i in scores_by_index]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)

        if self.diversity > 0:
            return self._apply_mmr(scored, top_k, self.diversity)
        return scored[:top_k]
