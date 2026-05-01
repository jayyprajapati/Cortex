from __future__ import annotations

from typing import List

from app.chunking.models import Chunk
from app.chunking.strategies.markdown_aware import MarkdownAwareStrategy
from app.chunking.strategies.resume_structured import ResumeStructuredStrategy
from app.chunking.strategies.semantic_doc import SemanticDocStrategy
from app.registry.models import IngestionConfig

_REGISTRY = {
    "semantic_doc": SemanticDocStrategy,
    "resume_structured": ResumeStructuredStrategy,
    "markdown_aware": MarkdownAwareStrategy,
}


def create_chunks(elements: List[dict], doc_id: str, config: IngestionConfig) -> List[Chunk]:
    strategy_class = _REGISTRY.get(config.strategy)
    if strategy_class is None:
        raise ValueError(
            f"Unknown chunking strategy: {config.strategy!r}. "
            f"Valid strategies: {sorted(_REGISTRY)}"
        )
    return strategy_class(config).chunk(elements, doc_id)
