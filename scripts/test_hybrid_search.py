"""Smoke test: hybrid search via ctx.components (doesn't require a running Qdrant)."""
from __future__ import annotations

from app.retrieval.fusion import reciprocal_rank_fusion, alpha_fusion
from app.retrieval.confidence import compute_confidence


def main():
    # Test RRF fusion
    list1 = [
        {"doc_id": "d1", "chunk_id": 0, "text": "apple pie", "score": 0.9},
        {"doc_id": "d2", "chunk_id": 1, "text": "banana bread", "score": 0.7},
    ]
    list2 = [
        {"doc_id": "d2", "chunk_id": 1, "text": "banana bread", "score": 0.85},
        {"doc_id": "d3", "chunk_id": 2, "text": "cherry cake", "score": 0.6},
    ]
    fused = reciprocal_rank_fusion([list1, list2])
    print(f"RRF fusion: {len(fused)} results")
    assert len(fused) == 3, f"Expected 3 unique results, got {len(fused)}"
    # d2 appears in both lists, should have highest RRF score
    assert fused[0]["doc_id"] == "d2", f"d2 should rank first, got {fused[0]['doc_id']}"

    # Test alpha fusion
    dense = [{"doc_id": "d1", "chunk_id": 0, "text": "a", "score": 0.9}]
    sparse = [{"doc_id": "d2", "chunk_id": 1, "text": "b", "score": 0.8}]
    alpha_result = alpha_fusion(dense, sparse, alpha=0.5)
    print(f"Alpha fusion: {len(alpha_result)} results")
    assert len(alpha_result) >= 1

    # Test confidence scoring
    high_chunks = [{"rerank_score": 0.85}]
    medium_chunks = [{"rerank_score": 0.4}]
    low_chunks = [{"score": 0.1}]
    empty = []

    assert compute_confidence(high_chunks) == "high"
    assert compute_confidence(medium_chunks) == "medium"
    assert compute_confidence(low_chunks) == "low"
    assert compute_confidence(empty) == "low"

    print("PASS: hybrid search/fusion tests passed")


if __name__ == "__main__":
    main()
