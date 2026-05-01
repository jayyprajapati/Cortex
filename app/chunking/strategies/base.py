from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.chunking.models import Chunk
from app.registry.models import IngestionConfig


class ChunkingStrategy(ABC):
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    @abstractmethod
    def chunk(self, elements: List[dict], doc_id: str) -> List[Chunk]:
        ...
