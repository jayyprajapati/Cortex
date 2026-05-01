from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.chunking.models import Chunk
from app.chunking.strategies.base import ChunkingStrategy
from app.chunking.tokenizer import token_count

_SECTION_RE = re.compile(
    r"^(?:"
    r"(?:work\s+)?experience|employment(?:\s+history)?|work\s+history|"
    r"professional\s+(?:background|experience)|"
    r"education(?:al\s+background)?|academic(?:\s+background)?|qualifications?|"
    r"(?:technical\s+)?skills?|core\s+competencies?|competencies?|"
    r"projects?|portfolio|"
    r"certifications?|licenses?|credentials?|"
    r"awards?|honors?|achievements?|accomplishments?|"
    r"publications?|research|"
    r"languages?|"
    r"volunteer(?:ing)?|community\s+(?:service|involvement)|"
    r"interests?|hobbies?|activities?|"
    r"(?:professional\s+)?summary|objective|profile|about(?:\s+me)?|"
    r"contact(?:\s+information)?|personal(?:\s+information)?|references?"
    r")$",
    re.IGNORECASE,
)


def _is_section_header(text: str) -> bool:
    text = text.strip()
    if not text or len(text.split()) > 6:
        return False
    return bool(_SECTION_RE.match(text.lower()))


class ResumeStructuredStrategy(ChunkingStrategy):
    """
    Resume/CV chunking that preserves section identity and creates
    discrete, independently meaningful chunks per section.
    No cross-section overlap — each section stands alone.
    """

    def chunk(self, elements: List[dict], doc_id: str) -> List[Chunk]:
        sections = self._group_sections(elements)
        chunks: List[Chunk] = []
        chunk_id = 0

        for section_name, page, lines in sections:
            for sc in self._split_section(lines, section_name):
                if sc["tokens"] < self.config.min_tokens:
                    continue
                chunks.append(
                    Chunk(
                        text=sc["text"],
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        section=section_name,
                        token_count=sc["tokens"],
                    )
                )
                chunk_id += 1

        return chunks

    def _group_sections(
        self, elements: List[dict]
    ) -> List[Tuple[str, Optional[int], List[str]]]:
        sections: List[Tuple[str, Optional[int], List[str]]] = []
        current_name: Optional[str] = None
        current_page: Optional[int] = None
        current_lines: List[str] = []

        for el in elements:
            text = str(el.get("text") or "").strip()
            page = el.get("page")
            if not text:
                continue

            if _is_section_header(text):
                if current_name is not None:
                    sections.append((current_name, current_page, current_lines))
                current_name = text.upper()
                current_page = page
                current_lines = []
            else:
                if current_name is None:
                    current_name = "HEADER"
                    current_page = page
                current_lines.append(text)

        if current_name is not None:
            sections.append((current_name, current_page, current_lines))

        return sections

    def _split_section(self, lines: List[str], section_name: str) -> List[dict]:
        max_tokens = self.config.max_tokens
        result: List[dict] = []
        header_tokens = token_count(section_name)
        current_parts: List[str] = [section_name]
        current_tokens = header_tokens

        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_tokens = token_count(line)

            if current_tokens + line_tokens > max_tokens and len(current_parts) > 1:
                result.append({"text": "\n".join(current_parts).strip(), "tokens": current_tokens})
                current_parts = [section_name, line]
                current_tokens = header_tokens + line_tokens
            else:
                current_parts.append(line)
                current_tokens += line_tokens

        if len(current_parts) > 1:
            result.append({"text": "\n".join(current_parts).strip(), "tokens": current_tokens})

        return result
