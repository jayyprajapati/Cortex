from __future__ import annotations

from typing import List, Optional

from app.chunking.base import BaseChunker
from app.chunking.models import Chunk
from app.chunking.tokenizer import token_count
from app.ingestion.loaders.base import Element


class LayoutAwareSemanticStrategy(BaseChunker):
    """
    Layout-aware semantic chunking strategy that preserves document structure.

    Key invariants:
    - section_path (stored in hierarchy) is NEVER None or empty — pre-heading
      content uses ["_root"].
    - Tables and code blocks are atomic — each becomes its own chunk.
    - No textual overlap — prev_chunk_id / next_chunk_id links handle continuity.
    - Small sections (< min_tokens) are merged with the previous sibling if they
      share the same hierarchy; heading boundaries are never crossed.
    """

    def __init__(self, config) -> None:
        # config needs: max_tokens, min_tokens
        # optional: keep_tables_atomic (default True), keep_code_atomic (default True)
        self.config = config

    def chunk(self, elements: List[Element], doc_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_id = 0

        heading_stack: List[str] = []          # current ancestor headings
        current_texts: List[str] = []
        current_tokens: int = 0
        current_page: Optional[int] = None
        current_element_types: List[str] = []

        def flush(force_section_path: Optional[List[str]] = None) -> None:
            nonlocal chunk_id, current_texts, current_tokens, current_page, current_element_types
            if not current_texts:
                return
            text = "\n\n".join(current_texts).strip()
            if not text:
                current_texts = []
                current_tokens = 0
                current_element_types = []
                return
            sp = force_section_path or (list(heading_stack) if heading_stack else ["_root"])
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                page=current_page,
                chunk_id=chunk_id,
                section=" > ".join(sp),
                hierarchy=sp,
                token_count=current_tokens,
            ))
            chunk_id += 1
            current_texts = []
            current_tokens = 0
            current_page = None
            current_element_types = []

        for el in elements:
            text = el.text.strip() if el.text else ""
            if not text:
                continue

            el_tokens = token_count(text)
            el_type = el.type or "paragraph"

            # Handle heading elements — flush current buffer and update heading_stack
            if el_type.startswith("heading_"):
                flush()
                level_str = el_type.replace("heading_l", "").replace("heading_", "")
                try:
                    level = int(level_str)
                except ValueError:
                    level = 1
                # Truncate heading_stack to the parent level, then push this heading
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(text)
                continue

            # Atomic blocks — always their own chunk
            keep_tables = getattr(self.config, "keep_tables_atomic", True)
            keep_code = getattr(self.config, "keep_code_atomic", True)
            is_atomic = (el_type == "table" and keep_tables) or (el_type == "code_block" and keep_code)

            if is_atomic:
                flush()
                sp = list(heading_stack) if heading_stack else ["_root"]
                chunks.append(Chunk(
                    text=text,
                    doc_id=doc_id,
                    page=el.page,
                    chunk_id=chunk_id,
                    section=" > ".join(sp),
                    hierarchy=sp,
                    token_count=el_tokens,
                ))
                chunk_id += 1
                continue

            # Normal text — pack into current buffer up to max_tokens
            if current_tokens > 0 and current_tokens + el_tokens > self.config.max_tokens:
                flush()
                # After flush, start fresh — no overlap needed; neighbor links handle continuity

            if current_page is None:
                current_page = el.page
            current_texts.append(text)
            current_tokens += el_tokens
            current_element_types.append(el_type)

        flush()  # flush any remaining text

        # Merge undersized chunks (< min_tokens) with their predecessor
        # Only merge if they share the same section path (same hierarchy)
        merged = _merge_small_chunks(chunks, self.config.min_tokens)

        # Renumber chunk_ids and set prev_chunk_id / next_chunk_id
        for i, c in enumerate(merged):
            c.chunk_id = i
            c.prev_chunk_id = merged[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = merged[i + 1].chunk_id if i < len(merged) - 1 else None

        return merged


def _merge_small_chunks(chunks: List[Chunk], min_tokens: int) -> List[Chunk]:
    """
    Merge any chunk below min_tokens into its predecessor, provided they share
    the same hierarchy (section path).  Heading boundaries are never crossed.
    """
    if not chunks:
        return chunks
    result: List[Chunk] = []
    for chunk in chunks:
        tokens = chunk.token_count or 0
        if (
            result
            and tokens < min_tokens
            and result[-1].hierarchy == chunk.hierarchy
        ):
            # Merge into predecessor
            prev = result[-1]
            prev.text = prev.text + "\n\n" + chunk.text
            prev.token_count = (prev.token_count or 0) + tokens
        else:
            result.append(chunk)
    return result
