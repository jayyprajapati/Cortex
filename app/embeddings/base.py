from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def supports_sparse(self) -> bool:
        return False

    @abstractmethod
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError("This embedder does not support sparse embeddings")
