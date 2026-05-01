from __future__ import annotations

import math
import re
from typing import List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """
    BM25 Okapi — pure Python implementation, no external dependencies.
    Designed for scoring a small candidate set (hundreds of documents)
    returned by dense retrieval, not a full corpus index.
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = [_tokenize(doc) for doc in documents]
        n = len(self._corpus)
        self._avgdl = sum(len(d) for d in self._corpus) / max(n, 1)
        self._idf: dict[str, float] = {}
        df: dict[str, int] = {}
        for doc in self._corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self._idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> List[float]:
        query_tokens = _tokenize(query)
        scores = [0.0] * len(self._corpus)
        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self._corpus):
                tf = doc.count(token)
                if tf == 0:
                    continue
                dl = len(doc)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-12)
        return scores
