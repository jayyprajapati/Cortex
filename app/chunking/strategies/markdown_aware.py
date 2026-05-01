from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.chunking.models import Chunk
from app.chunking.strategies.base import ChunkingStrategy
from app.chunking.tokenizer import token_count

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE_RE = re.compile(r"^```")


def _split_by_headers(text: str) -> List[dict]:
    """Split markdown text into sections delimited by ATX headers."""
    lines = text.splitlines()
    sections: List[dict] = []
    current_header: Optional[str] = None
    current_level = 0
    current_lines: List[str] = []
    in_fence = False

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence

        if not in_fence:
            m = _HEADER_RE.match(line)
            if m:
                if current_lines or current_header:
                    sections.append({
                        "header": current_header,
                        "level": current_level,
                        "content": "\n".join(current_lines),
                    })
                current_header = m.group(2).strip()
                current_level = len(m.group(1))
                current_lines = []
                continue

        current_lines.append(line)

    if current_lines or current_header:
        sections.append({
            "header": current_header,
            "level": current_level,
            "content": "\n".join(current_lines),
        })

    return sections


class MarkdownAwareStrategy(ChunkingStrategy):
    """
    Markdown-aware chunking that splits on ATX headers, preserves code fences
    intact, handles tables, and tracks heading hierarchy per chunk.
    """

    def chunk(self, elements: List[dict], doc_id: str) -> List[Chunk]:
        # Reassemble full markdown from elements
        full_text = "\n\n".join(
            str(el.get("text") or "").strip()
            for el in elements
            if str(el.get("text") or "").strip()
        )

        sections = _split_by_headers(full_text)
        chunks: List[Chunk] = []
        chunk_id = 0
        header_stack: List[Tuple[int, str]] = []

        for section in sections:
            header = section["header"]
            level = section["level"]
            content = section["content"].strip()

            if header:
                # Maintain ancestor stack
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, header))

            hierarchy = " > ".join(h for _, h in header_stack) if header_stack else None

            # Attempt to recover page number from original elements
            page = 1
            if header:
                for el in elements:
                    if header in str(el.get("text") or ""):
                        page = el.get("page") or 1
                        break

            if not content and not header:
                continue

            prefix = f"{'#' * level} {header}\n\n" if header else ""
            section_text = (prefix + content).strip()
            section_tokens = token_count(section_text)

            if section_tokens <= self.config.max_tokens:
                if section_tokens >= self.config.min_tokens:
                    chunks.append(
                        Chunk(
                            text=section_text,
                            doc_id=doc_id,
                            page=page,
                            chunk_id=chunk_id,
                            section=header,
                            hierarchy=hierarchy,
                            token_count=section_tokens,
                        )
                    )
                    chunk_id += 1
            else:
                for sc in self._split_large_section(section_text, header):
                    if sc["tokens"] >= self.config.min_tokens:
                        chunks.append(
                            Chunk(
                                text=sc["text"],
                                doc_id=doc_id,
                                page=page,
                                chunk_id=chunk_id,
                                section=header,
                                hierarchy=hierarchy,
                                token_count=sc["tokens"],
                            )
                        )
                        chunk_id += 1

        return chunks

    def _split_large_section(self, section_text: str, header: Optional[str]) -> List[dict]:
        max_tokens = self.config.max_tokens
        overlap_tokens = self.config.overlap_tokens
        paragraphs = re.split(r"\n\n+", section_text)
        result: List[dict] = []
        current_parts: List[str] = []
        current_tokens = 0
        overlap_tail = ""
        in_fence = False

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

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect code fence state
            fence_count = len(_FENCE_RE.findall(para))
            if fence_count % 2 != 0:
                in_fence = not in_fence

            para_tokens = token_count(para)

            # Never split code blocks
            if para.startswith("```") or in_fence:
                current_parts.append(para)
                current_tokens += para_tokens
                continue

            if current_tokens > 0 and current_tokens + para_tokens > max_tokens:
                flush()
                if overlap_tail:
                    carry_text = f"...{overlap_tail}"
                    current_parts.append(carry_text)
                    current_tokens += token_count(carry_text)

            current_parts.append(para)
            current_tokens += para_tokens

        flush()
        return result
