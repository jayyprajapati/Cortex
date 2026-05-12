"""Lightweight regex/heuristic extractor of document-level entity hints.

Pulls a small set of "identity" terms from the first non-empty elements of a
document — typically the candidate name, current title/role, and current employer
for a resume, or the document title for a paper/report.

These hints are stored on every chunk's Qdrant payload and concatenated to the
BM25 token stream at retrieval time, so questions phrased with synonyms or
identity terms ("who is the developer?", "who wrote this?") still surface
relevant chunks even when the literal query tokens don't appear in chunk text.
"""

from __future__ import annotations

import re
from typing import List

# How many leading elements form the document "header" used for entity extraction.
_HEADER_ELEMENT_LIMIT = 10
_HEADER_CHAR_LIMIT = 1500

# Title/role keywords that strongly suggest a job title or document role.
_ROLE_KEYWORDS = {
    "engineer", "developer", "designer", "manager", "analyst", "lead",
    "architect", "researcher", "scientist", "consultant", "specialist",
    "director", "founder", "co-founder", "intern", "associate",
    "administrator", "officer", "head", "principal", "senior", "junior",
    "staff", "vp", "cto", "ceo", "cfo", "coo", "pm", "sde", "swe",
    "programmer", "coder", "author", "writer", "editor",
}

_PROPER_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z'’]+(?:[ \t]+[A-Z][a-zA-Z'’]+){0,3})\b")
_TITLE_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z\- ]{2,80})\s*$"
)

_STOP_PROPER = {
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "I", "We", "You", "There", "Here", "Page", "Resume", "Curriculum",
    "Vitae", "CV", "PDF", "Email", "Phone", "Address",
}


def _gather_header_text(elements: List[dict]) -> str:
    parts: List[str] = []
    char_budget = _HEADER_CHAR_LIMIT
    for el in elements[:_HEADER_ELEMENT_LIMIT]:
        text = str((el or {}).get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        char_budget -= len(text)
        if char_budget <= 0:
            break
    return "\n".join(parts)


def _extract_proper_phrases(text: str) -> List[str]:
    out: List[str] = []
    for m in _PROPER_PHRASE_RE.finditer(text):
        phrase = m.group(1).strip()
        first = phrase.split()[0]
        if first in _STOP_PROPER:
            continue
        if phrase not in out:
            out.append(phrase)
    return out


def _extract_role_terms(text: str) -> List[str]:
    """Pull single-word and short multi-word role/title terms."""
    out: List[str] = []
    lowered = text.lower()
    # Single-word role keywords
    for kw in _ROLE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            if kw not in out:
                out.append(kw)
    # Common compound titles: "<adj> <role>"
    for m in re.finditer(r"\b([a-z]+)\s+(engineer|developer|designer|manager|analyst|architect|scientist|researcher|consultant|specialist)\b", lowered):
        compound = f"{m.group(1)} {m.group(2)}"
        if compound not in out:
            out.append(compound)
    return out


def extract_entity_hints(elements: List[dict]) -> List[str]:
    """Return up to ~15 entity hints derived from the document header.

    Best-effort and deliberately permissive — false positives are tolerable
    because hints only affect BM25 scoring, never chunk text shown to the LLM.
    """
    if not elements:
        return []

    header = _gather_header_text(elements)
    if not header:
        return []

    proper = _extract_proper_phrases(header)
    roles = _extract_role_terms(header)

    seen: set = set()
    hints: List[str] = []
    for item in proper + roles:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        hints.append(item)
        if len(hints) >= 15:
            break
    return hints
