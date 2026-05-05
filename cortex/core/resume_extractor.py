from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import hashlib

from app.ingestion.loader import load_document
from app.llm.factory import get_llm
from app.registry.service import _resolve_llm_config

# ---------------------------------------------------------------------------
# In-process extraction cache — keyed by sha256(raw_text):extraction_type
# Avoids re-running the LLM when the same file is uploaded more than once.
# ---------------------------------------------------------------------------

_EXTRACT_CACHE: Dict[str, Dict[str, Any]] = {}
_MAX_EXTRACT_CACHE = 50


def _extract_cache_key(raw_text: str, extraction_type: str) -> str:
    digest = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{digest}:{extraction_type}"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

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

_SECTION_ALIAS: Dict[str, str] = {
    "work experience": "experience",
    "employment history": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "professional background": "experience",
    "technical skills": "skills",
    "core competencies": "skills",
    "competencies": "skills",
    "academic background": "education",
    "educational background": "education",
    "qualifications": "education",
    "licenses": "certifications",
    "credentials": "certifications",
    "portfolio": "projects",
    "objective": "summary",
    "profile": "summary",
    "about me": "summary",
    "about": "summary",
    "contact information": "contact",
    "personal information": "contact",
}

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a structured career intelligence engine.

Extract ALL information from the document below and return ONLY valid JSON.

Rules:
- Return ONLY valid JSON. No markdown fences, prose, or comments outside the JSON object.
- Extraction type: {extraction_type}
- For each skill, include: name (string), category (string or null), proficiency (string or null).
- For each experience entry: company, title, date_range (string or null), location (string or null), bullets (array of strings).
- For each project: name, description (string), technologies (array of strings), url (string or null), date_range (string or null).
- For each education entry: institution, degree (string or null), field_of_study (string or null), date_range (string or null), gpa (string or null).
- certifications: array of plain strings.
- keywords: array of notable technologies, tools, and domain terms found anywhere in the document.
- raw_sections: object mapping normalized section name to the raw text content of that section.
- confidence: float 0.0-1.0 representing how complete and high-quality the extraction is.
- Use empty string "" for missing string fields, [] for missing arrays, null for missing optional scalars.

Required output schema (return exactly this shape):
{{
  "document_type": string,
  "skills": [{{"name": string, "category": string|null, "proficiency": string|null}}],
  "projects": [{{"name": string, "description": string, "technologies": [string], "url": string|null, "date_range": string|null}}],
  "experience": [{{"company": string, "title": string, "date_range": string|null, "location": string|null, "bullets": [string]}}],
  "education": [{{"institution": string, "degree": string|null, "field_of_study": string|null, "date_range": string|null, "gpa": string|null}}],
  "certifications": [string],
  "keywords": [string],
  "raw_sections": {{section_name: section_text}},
  "confidence": number
}}

Document text:
{text}

JSON Output:"""

_REQUIRED_KEYS = frozenset({
    "document_type", "skills", "projects", "experience",
    "education", "certifications", "keywords", "raw_sections",
})

_ARRAY_KEYS = ("skills", "projects", "experience", "education", "certifications", "keywords")

# Keywords that indicate an LLM provider-level error (auth, network, quota).
# These should not be retried — they will keep failing identically.
_LLM_AUTH_KEYWORDS = frozenset({
    "unauthorized", "forbidden", "authentication", "api key",
    "invalid token", "invalid key", " 401", " 403",
})


def _is_auth_error(exc: Exception) -> bool:
    """Return True if the exception looks like an LLM provider auth/permission failure."""
    msg = str(exc).strip().lower()
    return any(kw in msg for kw in _LLM_AUTH_KEYWORDS)


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _is_section_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 6:
        return False
    return bool(_SECTION_RE.match(stripped.lower()))


def _normalize_heading(heading: str) -> str:
    key = heading.strip().lower()
    return _SECTION_ALIAS.get(key, key)


def split_sections(text: str) -> Dict[str, str]:
    """Split plain text into a dict of {normalized_section_name: content}."""
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {}
    current = "header"
    sections[current] = []

    for line in lines:
        if _is_section_heading(line):
            current = _normalize_heading(line)
            if current not in sections:
                sections[current] = []
        else:
            stripped = line.strip()
            if stripped:
                sections[current].append(stripped)

    return {k: "\n".join(v) for k, v in sections.items() if v}


# ---------------------------------------------------------------------------
# JSON repair and validation
# ---------------------------------------------------------------------------

def _repair_json(raw: Any) -> Any:
    """Multi-pass JSON extraction: strip fences, extract from braces, validate."""
    stripped = str(raw).strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    candidates = [stripped]
    obj_s, obj_e = stripped.find("{"), stripped.rfind("}")
    if obj_s != -1 and obj_e > obj_s:
        candidates.append(stripped[obj_s: obj_e + 1])
    arr_s, arr_e = stripped.find("["), stripped.rfind("]")
    if arr_s != -1 and arr_e > arr_s:
        candidates.append(stripped[arr_s: arr_e + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Cannot parse JSON from model output. Preview: {stripped[:400]!r}"
    )


def _validate_extract_payload(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root, got {type(data).__name__}")

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    for key in _ARRAY_KEYS:
        if not isinstance(data[key], list):
            raise ValueError(f"'{key}' must be an array, got {type(data[key]).__name__}")

    if not isinstance(data.get("raw_sections"), dict):
        raise ValueError(
            f"'raw_sections' must be an object, got {type(data.get('raw_sections')).__name__}"
        )


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------

def extract_resume(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    doc_id: Optional[str] = None,
    extraction_type: str = "resume",
    llm_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Structured extraction from a resume, profile document, or raw text.

    Accepts exactly one of `text` or `file_path`. Returns a dict that
    conforms to the ExtractResponse schema in cortex.schemas.resumelab.

    Args:
        text: Raw document text.
        file_path: Path to a PDF, DOCX, or text file.
        doc_id: Caller-supplied document identifier; auto-generated if omitted.
        extraction_type: "resume" | "generic_profile" | "structured_doc"
        llm_override: Optional LLM provider dict, same shape as /extract request.

    Raises:
        ValueError: On empty input, parse failure, or schema violation after retries.
    """
    resolved_doc_id = str(doc_id).strip() if doc_id else str(uuid.uuid4())
    source_type = "text"

    if file_path and text:
        raise ValueError("Provide either text or file_path, not both.")

    if file_path:
        elements = load_document(file_path)
        raw_text = "\n".join(
            str(el.get("text") or "").strip()
            for el in elements
            if el.get("text")
        )
        ext = os.path.splitext(file_path)[1].lower()
        source_type = (
            "pdf" if ext == ".pdf"
            else "docx" if ext in (".docx", ".doc")
            else "file"
        )
    elif text:
        raw_text = str(text).strip()
        source_type = "text"
    else:
        raise ValueError("Provide either text or file_path for extraction.")

    if not raw_text.strip():
        raise ValueError("Document produced no extractable text.")

    fallback_sections = split_sections(raw_text)

    cache_key = _extract_cache_key(raw_text, extraction_type)
    if cache_key in _EXTRACT_CACHE:
        cached = _EXTRACT_CACHE[cache_key]
        logger.info("resume_extractor cache hit doc_id=%s", resolved_doc_id)
        return {
            **cached,
            "doc_id": resolved_doc_id,
            "metadata": {
                **cached["metadata"],
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "cache_hit": True,
            },
        }

    llm_config = _resolve_llm_config(llm_override)
    llm = get_llm(llm_config)

    prompt = _EXTRACT_PROMPT.format(
        extraction_type=extraction_type,
        text=raw_text[:8000],
    )

    max_retries = 3
    last_error: Optional[Exception] = None
    parsed: Optional[Dict[str, Any]] = None

    for attempt in range(max_retries + 1):
        try:
            output = llm.generate(prompt, temperature=0.05)
            parsed = _repair_json(output)
            _validate_extract_payload(parsed)
            logger.info(
                "resume_extractor success attempt=%d/%d doc_id=%s",
                attempt + 1, max_retries + 1, resolved_doc_id,
            )
            break
        except Exception as exc:
            # Auth/permission errors will keep failing — skip retries immediately.
            if _is_auth_error(exc):
                raise ValueError(
                    f"LLM provider authentication failed: {exc}. "
                    "Verify your API key in the environment configuration."
                ) from exc
            last_error = exc
            logger.warning(
                "resume_extractor attempt %d/%d failed: %s",
                attempt + 1, max_retries + 1, exc,
            )
            if attempt < max_retries:
                prompt = (
                    f"{prompt}\n\n"
                    "CORRECTION: Your previous response was invalid. "
                    "Return ONLY valid JSON matching the required schema exactly. "
                    "No markdown fences, no prose outside the JSON object."
                )

    if parsed is None:
        raise ValueError(
            f"LLM extraction failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    raw_sections = parsed.get("raw_sections") or fallback_sections
    if not raw_sections:
        raw_sections = fallback_sections

    confidence_raw = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    result = {
        "doc_id": resolved_doc_id,
        "document_type": str(parsed.get("document_type") or extraction_type),
        "skills": parsed.get("skills") or [],
        "projects": parsed.get("projects") or [],
        "experience": parsed.get("experience") or [],
        "education": parsed.get("education") or [],
        "certifications": parsed.get("certifications") or [],
        "keywords": parsed.get("keywords") or [],
        "raw_sections": raw_sections,
        "metadata": {
            "source_type": source_type,
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
        },
        # A1: Source-preservation fields — derived from the raw text, no extra LLM call.
        "normalized_resume_text": raw_text.strip(),
        "sectioned_resume_source": raw_sections,
    }

    if len(_EXTRACT_CACHE) >= _MAX_EXTRACT_CACHE:
        oldest = next(iter(_EXTRACT_CACHE))
        del _EXTRACT_CACHE[oldest]
    _EXTRACT_CACHE[cache_key] = result
    return result
