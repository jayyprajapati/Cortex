from dataclasses import dataclass
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
    # Resume-canonical fields — populated by ResumeCanonicalStrategy and ingest pipeline.
    # Ignored by all other strategies; backward-compatible additions.
    canonical_type: Optional[str] = None   # skill | experience | project | education | certification | summary | contact | misc
    canonical_key: Optional[str] = None    # free-form identifier (e.g. "Google | Senior Engineer")
    source_section: Optional[str] = None  # original heading text before normalization
    source_app: Optional[str] = None      # app_name from ExecutionContext, set by ingest pipeline
