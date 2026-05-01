from __future__ import annotations

from typing import List, Optional

from app.chunking.models import Chunk
from app.chunking.strategies.base import ChunkingStrategy
from app.chunking.tokenizer import token_count

_JUNK_PREFIXES = ("sample document content", "title:", "---", "===")
_HEADING_TERMINAL_PUNCT = (".", "!", "?", ":", ";")


def _is_heading(text: str, max_words: int = 12) -> bool:
    text = text.strip()
    if not text or text.endswith(_HEADING_TERMINAL_PUNCT):
        return False
    words = text.split()
    if len(words) > max_words:
        return False
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized >= max(1, len(words) // 2)


def _is_junk(text: str) -> bool:
    lower = text.strip().lower()
    # Only filter explicit noise patterns; never filter by word count alone
    # (a 1-word heading like "Introduction" is valid content)
    return not lower or any(lower.startswith(p) for p in _JUNK_PREFIXES)


class SemanticDocStrategy(ChunkingStrategy):
    """
    Production document chunking with heading-hierarchy detection,
    token-bounded sections, and configurable overlap carry-forward.
    """

    def chunk(self, elements: List[dict], doc_id: str) -> List[Chunk]:
        sections = self._group_sections(elements)
        chunks: List[Chunk] = []
        chunk_id = 0

        for section in sections:
            heading = section["heading"]
            page = section["page"]
            section_chunks = self._split_section(
                paragraphs=section["paragraphs"],
                heading=heading,
            )

            # Merge undersized tail chunk into its predecessor
            if len(section_chunks) > 1:
                tail = section_chunks[-1]
                if tail["tokens"] < self.config.min_tokens:
                    section_chunks[-2]["text"] += "\n\n" + tail["text"]
                    section_chunks[-2]["tokens"] += tail["tokens"]
                    section_chunks.pop()

            for sc in section_chunks:
                if sc["tokens"] < self.config.min_tokens:
                    continue
                chunks.append(
                    Chunk(
                        text=sc["text"],
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        section=heading,
                        hierarchy=sc.get("hierarchy"),
                        token_count=sc["tokens"],
                    )
                )
                chunk_id += 1

        return chunks

    def _group_sections(self, elements: List[dict]) -> List[dict]:
        sections: List[dict] = []
        current: Optional[dict] = None

        for el in elements:
            text = str(el.get("text") or "").strip()
            page = el.get("page")
            if not text or _is_junk(text):
                continue

            if _is_heading(text):
                if current:
                    sections.append(current)
                current = {"heading": text, "page": page, "paragraphs": []}
                continue

            if current is None:
                current = {"heading": None, "page": page, "paragraphs": []}

            current["paragraphs"].append(text)

        if current:
            sections.append(current)

        return sections

    def _split_section(self, paragraphs: List[str], heading: Optional[str]) -> List[dict]:
        max_tokens = self.config.max_tokens
        overlap_tokens = self.config.overlap_tokens
        heading_tokens = token_count(heading) if heading else 0
        result: List[dict] = []
        current_parts: List[str] = []
        current_tokens = 0
        overlap_tail = ""

        def flush() -> None:
            nonlocal current_parts, current_tokens, overlap_tail
            if not current_parts:
                return
            text = "\n\n".join(current_parts).strip()
            if text:
                result.append({"text": text, "tokens": current_tokens})
            if overlap_tokens > 0:
                words = text.split()
                carry = max(1, overlap_tokens * 3 // 4)
                overlap_tail = " ".join(words[-carry:])
            current_parts = []
            current_tokens = 0

        if heading:
            current_parts.append(heading)
            current_tokens += heading_tokens

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = token_count(para)

            if current_tokens > 0 and current_tokens + para_tokens > max_tokens:
                flush()
                if heading:
                    current_parts.append(heading)
                    current_tokens += heading_tokens
                if overlap_tail:
                    carry_text = f"[...continued] {overlap_tail}"
                    current_parts.append(carry_text)
                    current_tokens += token_count(carry_text)

            current_parts.append(para)
            current_tokens += para_tokens

        flush()
        return result
