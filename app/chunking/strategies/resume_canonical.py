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

_CANONICAL_TYPE_MAP: dict = {
    "experience": "experience",
    "work experience": "experience",
    "employment history": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "professional background": "experience",
    "skills": "skill",
    "technical skills": "skill",
    "core competencies": "skill",
    "competencies": "skill",
    "projects": "project",
    "portfolio": "project",
    "education": "education",
    "academic background": "education",
    "educational background": "education",
    "qualifications": "education",
    "certifications": "certification",
    "licenses": "certification",
    "credentials": "certification",
    "awards": "certification",
    "honors": "certification",
    "achievements": "certification",
    "accomplishments": "certification",
    "summary": "summary",
    "objective": "summary",
    "profile": "summary",
    "about": "summary",
    "about me": "summary",
    "contact": "contact",
    "contact information": "contact",
    "personal information": "contact",
    "languages": "skill",
    "publications": "project",
    "research": "project",
}


def _is_section_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped.split()) > 6:
        return False
    return bool(_SECTION_RE.match(stripped.lower()))


def _canonical_type(heading: str) -> str:
    return _CANONICAL_TYPE_MAP.get(heading.strip().lower(), "misc")


class ResumeCanonicalStrategy(ChunkingStrategy):
    """
    Resume ingestion strategy that produces semantically typed chunks.

    Each chunk carries canonical_type (skill/experience/project/education/
    certification/summary/contact/misc) and source_section (original heading
    text) to enable fine-grained filtered retrieval from Qdrant.

    No cross-section overlap — each section is a self-contained unit.
    """

    def chunk(self, elements: List[dict], doc_id: str) -> List[Chunk]:
        sections = self._group_sections(elements)
        chunks: List[Chunk] = []
        chunk_id = 0

        for section_heading, ctype, page, lines in sections:
            for sc in self._split_section(lines, section_heading):
                if sc["tokens"] < self.config.min_tokens:
                    continue
                chunks.append(
                    Chunk(
                        text=sc["text"],
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        section=section_heading,
                        token_count=sc["tokens"],
                        canonical_type=ctype,
                        source_section=section_heading,
                    )
                )
                chunk_id += 1

        return chunks

    def _group_sections(
        self, elements: List[dict]
    ) -> List[Tuple[str, str, Optional[int], List[str]]]:
        """
        Returns list of (heading, canonical_type, page, lines).
        Pre-header content goes into a synthetic "HEADER" section.
        """
        sections: List[Tuple[str, str, Optional[int], List[str]]] = []
        current_heading: Optional[str] = None
        current_ctype: str = "misc"
        current_page: Optional[int] = None
        current_lines: List[str] = []

        for el in elements:
            text = str(el.get("text") or "").strip()
            page = el.get("page")
            if not text:
                continue

            if _is_section_heading(text):
                if current_heading is not None:
                    sections.append((current_heading, current_ctype, current_page, current_lines))
                current_heading = text.upper()
                current_ctype = _canonical_type(text)
                current_page = page
                current_lines = []
            else:
                if current_heading is None:
                    current_heading = "HEADER"
                    current_ctype = "misc"
                    current_page = page
                current_lines.append(text)

        if current_heading is not None:
            sections.append((current_heading, current_ctype, current_page, current_lines))

        return sections

    def _split_section(self, lines: List[str], section_heading: str) -> List[dict]:
        """
        Split section lines into token-budget chunks, each prefixed with the
        section heading so the chunk is independently interpretable.
        """
        max_tokens = self.config.max_tokens
        header_tokens = token_count(section_heading)
        result: List[dict] = []
        current_parts: List[str] = [section_heading]
        current_tokens = header_tokens

        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_tokens = token_count(line)

            if current_tokens + line_tokens > max_tokens and len(current_parts) > 1:
                result.append({
                    "text": "\n".join(current_parts).strip(),
                    "tokens": current_tokens,
                })
                current_parts = [section_heading, line]
                current_tokens = header_tokens + line_tokens
            else:
                current_parts.append(line)
                current_tokens += line_tokens

        if len(current_parts) > 1:
            result.append({
                "text": "\n".join(current_parts).strip(),
                "tokens": current_tokens,
            })

        return result
