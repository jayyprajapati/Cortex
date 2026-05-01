from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SearchResult:
    text: str
    doc_id: str
    page: Optional[int]
    section: Optional[str]
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None
    hierarchy: Optional[str] = None
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "page": self.page,
            "section": self.section,
            "score": self.score,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
            "hierarchy": self.hierarchy,
            "token_count": self.token_count,
        }
