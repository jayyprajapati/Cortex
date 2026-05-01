from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingResult:
    vectors: List[list]
    model: str
    dimension: int
    count: int
