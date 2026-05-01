from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    text: str
    doc_id: str
    page: Optional[int]
    chunk_id: int
    section: Optional[str] = None
    hierarchy: Optional[str] = None
    token_count: Optional[int] = None
