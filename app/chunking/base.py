from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.chunking.models import Chunk
from app.ingestion.loaders.base import Element


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, elements: List[Element], doc_id: str) -> List[Chunk]:
        ...
