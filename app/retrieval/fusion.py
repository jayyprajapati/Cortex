from __future__ import annotations

from typing import Dict, List, Tuple


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    key_fn=None,
    k: int = 60,
) -> List[Dict]:
    """
    Merge multiple ranked result lists using RRF.
    key_fn: function to extract a dedup key from a result dict. Defaults to chunk_id or text fingerprint.
    Returns merged list sorted by RRF score descending.
    """
    if not result_lists:
        return []

    def _default_key(chunk: Dict) -> str:
        doc_id = str(chunk.get("doc_id") or "")
        chunk_id = chunk.get("chunk_id")
        if chunk_id is not None:
            return f"{doc_id}:{chunk_id}"
        text = " ".join(str(chunk.get("text") or "").split())[:80]
        return f"{doc_id}:{text}"

    kf = key_fn or _default_key
    scores: Dict[str, float] = {}
    by_key: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            key = kf(chunk)
            scores[key] = scores.get(key, 0.0) + rrf_score(rank, k)
            if key not in by_key or chunk.get("score", 0.0) > by_key[key].get("score", 0.0):
                by_key[key] = chunk

    result = list(by_key.values())
    result.sort(key=lambda c: scores[kf(c)], reverse=True)
    for chunk in result:
        chunk["rrf_score"] = scores[kf(chunk)]
    return result


def alpha_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    alpha: float = 0.5,
) -> List[Dict]:
    """
    Linear interpolation of normalized dense and sparse scores.
    alpha=1.0 → pure dense. alpha=0.0 → pure sparse.
    """
    def _normalize(items: List[Dict], score_key: str = "score") -> Dict[str, float]:
        scores = [c.get(score_key, 0.0) for c in items]
        if not scores:
            return {}
        lo, hi = min(scores), max(scores)
        span = hi - lo

        def _default_key(chunk: Dict) -> str:
            doc_id = str(chunk.get("doc_id") or "")
            chunk_id = chunk.get("chunk_id")
            if chunk_id is not None:
                return f"{doc_id}:{chunk_id}"
            text = " ".join(str(chunk.get("text") or "").split())[:80]
            return f"{doc_id}:{text}"

        return {
            _default_key(c): (c.get(score_key, 0.0) - lo) / span if span > 1e-9 else 1.0
            for c in items
        }

    dense_norm = _normalize(dense_results)
    sparse_norm = _normalize(sparse_results)
    all_keys = set(dense_norm) | set(sparse_norm)

    by_key: Dict[str, Dict] = {}
    for r in dense_results + sparse_results:
        doc_id = str(r.get("doc_id") or "")
        chunk_id = r.get("chunk_id")
        if chunk_id is not None:
            key = f"{doc_id}:{chunk_id}"
        else:
            text = " ".join(str(r.get("text") or "").split())[:80]
            key = f"{doc_id}:{text}"
        if key not in by_key:
            by_key[key] = r

    result = []
    for key in all_keys:
        if key not in by_key:
            continue
        chunk = dict(by_key[key])
        d = dense_norm.get(key, 0.0)
        s = sparse_norm.get(key, 0.0)
        chunk["score"] = alpha * d + (1.0 - alpha) * s
        result.append(chunk)

    result.sort(key=lambda c: c["score"], reverse=True)
    return result
