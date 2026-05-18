from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    doc_id: str
    page: Optional[int]
    chunk_id: int
    section: Optional[str] = None
    hierarchy: Optional[List[str]] = None
    token_count: Optional[int] = None
    prev_chunk_id: Optional[int] = None
    next_chunk_id: Optional[int] = None
    # Resume-canonical fields — populated by ResumeCanonicalStrategy and ingest pipeline.
    # Ignored by all other strategies; backward-compatible additions.
    canonical_type: Optional[str] = None   # skill | experience | project | education | certification | summary | contact | misc
    canonical_key: Optional[str] = None    # free-form identifier (e.g. "Google | Senior Engineer")
    source_section: Optional[str] = None  # original heading text before normalization
    source_app: Optional[str] = None      # app_name from ExecutionContext, set by ingest pipeline
    # Document-level entity hints (candidate name, title, role, company) — populated by
    # the ingest pipeline. Appended to the BM25 token stream at retrieval time to boost
    # recall on synonym/identity questions (e.g., "who is the developer" → matches
    # chunks even when the literal token "developer" is absent from the text).
    entity_hints: List[str] = field(default_factory=list)
