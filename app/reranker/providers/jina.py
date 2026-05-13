from __future__ import annotations

from typing import Any, Dict, List

from app.reranker.base import BaseReranker


class JinaReranker(BaseReranker):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.model_name = options.get("model", "jina-reranker-v2-base-multilingual")
        self.api_key = options.get("api_key")
        self.diversity = options.get("diversity", 0.0)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        candidate_cap: int,
    ) -> List[Dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is not installed. Run: pip install httpx")

        candidates = chunks[:candidate_cap]
        if not candidates:
            return []

        response = httpx.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": query,
                "documents": [c["text"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        for r in results:
            idx = r["index"]
            candidates[idx]["rerank_score"] = r["relevance_score"]
            candidates[idx]["rerank_raw_score"] = r["relevance_score"]

        scored = [
            candidates[r["index"]]
            for r in sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        ]
        return scored[:top_k]
