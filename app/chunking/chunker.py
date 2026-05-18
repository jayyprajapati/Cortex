from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from app.chunking.models import Chunk
from app.chunking.strategies.layout_aware_semantic import LayoutAwareSemanticStrategy
from app.chunking.strategies.markdown_aware import MarkdownAwareStrategy
from app.chunking.strategies.resume_canonical import ResumeCanonicalStrategy
from app.chunking.strategies.resume_structured import ResumeStructuredStrategy
from app.chunking.strategies.semantic_doc import SemanticDocStrategy
from app.registry.models import IngestionConfig

_REGISTRY = {
    "semantic_doc": SemanticDocStrategy,
    "resume_structured": ResumeStructuredStrategy,
    "markdown_aware": MarkdownAwareStrategy,
    "resume_canonical": ResumeCanonicalStrategy,
    "layout_aware_semantic": LayoutAwareSemanticStrategy,
}

# Strategies that take List[Element] rather than List[dict]
_ELEMENT_BASED_STRATEGIES = {"layout_aware_semantic"}


def _dicts_to_elements(elements: List[dict]):
    from app.ingestion.loaders.base import Element
    result = []
    for el in elements:
        text = str(el.get("text") or "").strip()
        if not text:
            continue
        el_type = el.get("type") or "paragraph"
        result.append(Element(
            type=el_type,
            text=text,
            page=el.get("page"),
            bbox=None,
            metadata={},
            parent_heading=el.get("section"),
        ))
    return result


def _resolve_chunker(config) -> "BaseChunker":
    """Instantiate a chunker from a ChunkingConfig (or any object with .strategy)."""
    from app.chunking.base import BaseChunker
    strategy_name = getattr(config, "strategy", "layout_aware_semantic")
    strategy_class = _REGISTRY.get(strategy_name)
    if strategy_class is None:
        raise ValueError(
            f"Unknown chunking strategy: {strategy_name!r}. "
            f"Valid: {sorted(_REGISTRY)}"
        )
    return strategy_class(config)


def create_chunks(elements: Union[List[dict], list], doc_id: str, config: IngestionConfig) -> List[Chunk]:
    strategy_class = _REGISTRY.get(config.strategy)
    if strategy_class is None:
        raise ValueError(
            f"Unknown chunking strategy: {config.strategy!r}. "
            f"Valid strategies: {sorted(_REGISTRY)}"
        )

    if config.strategy in _ELEMENT_BASED_STRATEGIES:
        # Convert plain dicts to Element objects if the caller passed dicts
        from app.ingestion.loaders.base import Element as _Element
        if elements and not isinstance(elements[0], _Element):
            elements = _dicts_to_elements(elements)
        return strategy_class(config).chunk(elements, doc_id)

    # Legacy strategies expect List[dict]
    return strategy_class(config).chunk(elements, doc_id)
