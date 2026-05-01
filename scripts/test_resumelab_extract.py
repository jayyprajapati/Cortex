"""
ResumeLab extraction test suite.

Run with:
    python -m scripts.test_resumelab_extract

Each test case exercises extract_resume() directly (no HTTP layer).
Tests are self-contained — no Qdrant or LLM required for structural/unit cases;
LLM-dependent cases are marked and can be skipped with --no-llm.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal test harness
# ---------------------------------------------------------------------------

_results: List[Dict[str, Any]] = []


def _run(name: str, fn: Callable, skip: bool = False) -> None:
    if skip:
        _results.append({"name": name, "status": "SKIP"})
        print(f"  SKIP  {name}")
        return
    try:
        fn()
        _results.append({"name": name, "status": "PASS"})
        print(f"  PASS  {name}")
    except AssertionError as exc:
        _results.append({"name": name, "status": "FAIL", "error": str(exc)})
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:
        _results.append({"name": name, "status": "ERROR", "error": str(exc)})
        print(f"  ERROR {name}: {exc}")
        traceback.print_exc()


def _assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg} | expected {expected!r}, got {actual!r}")


def _assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{msg} | {item!r} not in {container!r}")


def _assert_type(value, t, msg=""):
    if not isinstance(value, t):
        raise AssertionError(f"{msg} | expected {t.__name__}, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLEAN_RESUME = """
Jane Smith
jane@example.com | linkedin.com/in/janesmith | (555) 123-4567

Summary
Results-driven software engineer with 7 years of experience building scalable systems.

Experience
Google – Senior Software Engineer | 2020 – Present | Mountain View, CA
- Led migration of monolith to microservices, reducing P95 latency by 40%
- Mentored 4 junior engineers and drove quarterly OKR planning
- Built internal tooling with Python and Go that saved 200 engineer-hours/month

Meta – Software Engineer | 2017 – 2020 | Menlo Park, CA
- Developed real-time data pipeline processing 50M events/day using Kafka and Flink
- Reduced deployment cycle from 2 weeks to 2 days via CI/CD automation

Skills
Python, Go, Java, Kubernetes, Docker, Kafka, Flink, PostgreSQL, Redis, AWS, GCP

Education
Stanford University | B.S. Computer Science | 2017 | GPA: 3.9

Certifications
AWS Certified Solutions Architect – Associate
Google Cloud Professional Data Engineer

Projects
DataFlow – Open-source ETL framework
- Built a distributed ETL framework used by 500+ teams
- Technologies: Python, Apache Spark, Airflow
- github.com/janesmith/dataflow
"""

MESSY_RESUME = """
MESSY RESUME WITH DUPLICATES AND WEIRD FORMATTING

SKILLS
Python JavaScript React
Node.js PostgreSQL

SKILLS
Also knows: Docker, AWS

EXPERIENCE

Some Company – Developer 2019-2021
Did stuff with databases and APIs

EXPERIENCE
Another Firm – Senior Dev – 2021-Now
Builds microservices

Education
State University BS CS 2019

certifications
AWS CCP
"""

MISSING_HEADINGS_RESUME = """
John Doe
john.doe@example.com

Experienced backend developer. 8 years in fintech, specializing in payment systems.

Acme Corp, Lead Engineer, 2018-present
Designed and implemented payment gateway handling $5B/day in transactions.
Built fraud detection system reducing chargebacks by 30%.

StartupXYZ, Engineer, 2015-2018
Developed REST APIs for mobile banking app (2M users).
Worked with Node.js, PostgreSQL, Redis.
"""

JD_LIKE_TEXT = """
Software Engineer – Backend

We are looking for a Senior Backend Engineer to join our team.

Requirements:
- 5+ years of Python or Go experience
- Experience with microservices architecture
- Strong knowledge of PostgreSQL and Redis
- Experience with AWS or GCP
- Familiarity with Kubernetes and Docker

Nice to have:
- Experience with Kafka or RabbitMQ
- Knowledge of ML pipelines
- Open-source contributions

Responsibilities:
- Design and implement scalable backend services
- Collaborate with frontend and ML teams
- Lead code reviews and mentor junior engineers
"""

_MOCK_VALID_RESPONSE = json.dumps({
    "document_type": "resume",
    "skills": [
        {"name": "Python", "category": "programming", "proficiency": "expert"},
        {"name": "Go", "category": "programming", "proficiency": "proficient"},
    ],
    "projects": [
        {
            "name": "DataFlow",
            "description": "Open-source ETL framework",
            "technologies": ["Python", "Apache Spark", "Airflow"],
            "url": "github.com/janesmith/dataflow",
            "date_range": None,
        }
    ],
    "experience": [
        {
            "company": "Google",
            "title": "Senior Software Engineer",
            "date_range": "2020 – Present",
            "location": "Mountain View, CA",
            "bullets": [
                "Led migration of monolith to microservices",
                "Mentored 4 junior engineers",
            ],
        }
    ],
    "education": [
        {
            "institution": "Stanford University",
            "degree": "B.S.",
            "field_of_study": "Computer Science",
            "date_range": "2017",
            "gpa": "3.9",
        }
    ],
    "certifications": [
        "AWS Certified Solutions Architect – Associate",
        "Google Cloud Professional Data Engineer",
    ],
    "keywords": ["Python", "Go", "Kubernetes", "Kafka", "AWS"],
    "raw_sections": {"experience": "Google – Senior...", "skills": "Python, Go..."},
    "confidence": 0.92,
})

_MOCK_MALFORMED_FIRST = "Here is the JSON: ```json\n" + _MOCK_VALID_RESPONSE + "\n```"

_MOCK_MISSING_KEY = json.dumps({
    "document_type": "resume",
    "skills": [],
    "projects": [],
    # missing experience, education, certifications, keywords, raw_sections
})


# ---------------------------------------------------------------------------
# Unit tests: section splitter (no LLM)
# ---------------------------------------------------------------------------

def test_section_splitter_clean():
    from cortex.core.resume_extractor import split_sections
    sections = split_sections(CLEAN_RESUME)
    assert "experience" in sections, f"Missing 'experience'. Keys: {list(sections)}"
    assert "skills" in sections, f"Missing 'skills'. Keys: {list(sections)}"
    assert "education" in sections, f"Missing 'education'. Keys: {list(sections)}"
    assert "certifications" in sections, f"Missing 'certifications'. Keys: {list(sections)}"
    assert "projects" in sections, f"Missing 'projects'. Keys: {list(sections)}"


def test_section_splitter_alias_normalization():
    from cortex.core.resume_extractor import split_sections
    text = "Technical Skills\nPython, Java\n\nWork Experience\nGoogle 2020-2024"
    sections = split_sections(text)
    assert "skills" in sections, f"'technical skills' should normalize to 'skills'. Got: {list(sections)}"
    assert "experience" in sections, f"'work experience' should normalize to 'experience'. Got: {list(sections)}"


def test_section_splitter_missing_headings():
    from cortex.core.resume_extractor import split_sections
    sections = split_sections(MISSING_HEADINGS_RESUME)
    # Should produce a 'header' section with all the content
    assert len(sections) > 0
    assert any(v for v in sections.values()), "All sections are empty"


def test_section_splitter_empty_input():
    from cortex.core.resume_extractor import split_sections
    sections = split_sections("")
    assert sections == {} or all(not v for v in sections.values()), \
        "Empty input should produce empty sections"


def test_section_splitter_duplicate_headings():
    from cortex.core.resume_extractor import split_sections
    sections = split_sections(MESSY_RESUME)
    # Second 'skills' block should merge or be handled gracefully — must not crash
    assert isinstance(sections, dict)


# ---------------------------------------------------------------------------
# Unit tests: JSON repair (no LLM)
# ---------------------------------------------------------------------------

def test_json_repair_clean_object():
    from cortex.core.resume_extractor import _repair_json
    data = _repair_json('{"key": "value"}')
    assert data == {"key": "value"}


def test_json_repair_strips_markdown_fence():
    from cortex.core.resume_extractor import _repair_json
    data = _repair_json('```json\n{"key": "value"}\n```')
    assert data == {"key": "value"}


def test_json_repair_extracts_from_prose():
    from cortex.core.resume_extractor import _repair_json
    data = _repair_json('Here is your JSON: {"answer": 42} — enjoy!')
    assert data == {"answer": 42}


def test_json_repair_raises_on_garbage():
    from cortex.core.resume_extractor import _repair_json
    try:
        _repair_json("this is not json at all")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Unit tests: schema validator (no LLM)
# ---------------------------------------------------------------------------

def test_validate_extract_payload_valid():
    from cortex.core.resume_extractor import _validate_extract_payload
    import json
    data = json.loads(_MOCK_VALID_RESPONSE)
    _validate_extract_payload(data)  # should not raise


def test_validate_extract_payload_missing_key():
    from cortex.core.resume_extractor import _validate_extract_payload
    import json
    data = json.loads(_MOCK_MISSING_KEY)
    try:
        _validate_extract_payload(data)
        raise AssertionError("Should have raised ValueError for missing keys")
    except ValueError as exc:
        assert "Missing required keys" in str(exc)


def test_validate_extract_payload_wrong_type():
    from cortex.core.resume_extractor import _validate_extract_payload
    data = {
        "document_type": "resume",
        "skills": "not-a-list",  # wrong type
        "projects": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "keywords": [],
        "raw_sections": {},
    }
    try:
        _validate_extract_payload(data)
        raise AssertionError("Should have raised ValueError for wrong type")
    except ValueError as exc:
        assert "skills" in str(exc)


# ---------------------------------------------------------------------------
# Unit tests: ResumeCanonicalStrategy (no LLM)
# ---------------------------------------------------------------------------

def test_resume_canonical_strategy_clean():
    from app.chunking.strategies.resume_canonical import ResumeCanonicalStrategy
    from app.registry.models import IngestionConfig
    config = IngestionConfig(
        strategy="resume_canonical",
        max_tokens=512,
        min_tokens=10,
        overlap_tokens=0,
        semantic_split=False,
    )
    strategy = ResumeCanonicalStrategy(config)
    elements = [
        {"text": "Jane Smith", "page": 1},
        {"text": "Summary", "page": 1},
        {"text": "Experienced engineer with 7 years building scalable distributed systems.", "page": 1},
        {"text": "Skills", "page": 1},
        {"text": "Python, Go, Kubernetes, Docker, PostgreSQL, Redis, AWS, Kafka, Flink, Terraform", "page": 1},
        {"text": "Experience", "page": 1},
        {"text": "Google – Senior Engineer 2020-Present", "page": 1},
        {"text": "Led microservices migration reducing P95 latency by 40%", "page": 1},
    ]
    chunks = strategy.chunk(elements, "test-doc-001")
    assert len(chunks) > 0, "Should produce at least one chunk"

    ctypes = {c.canonical_type for c in chunks}
    assert "skill" in ctypes, f"Expected 'skill' canonical_type. Got: {ctypes}"
    assert "experience" in ctypes, f"Expected 'experience' canonical_type. Got: {ctypes}"

    for chunk in chunks:
        assert chunk.doc_id == "test-doc-001"
        assert chunk.canonical_type is not None
        assert chunk.source_section is not None


def test_resume_canonical_strategy_empty_elements():
    from app.chunking.strategies.resume_canonical import ResumeCanonicalStrategy
    from app.registry.models import IngestionConfig
    config = IngestionConfig(
        strategy="resume_canonical",
        max_tokens=512,
        min_tokens=10,
        overlap_tokens=0,
        semantic_split=False,
    )
    strategy = ResumeCanonicalStrategy(config)
    chunks = strategy.chunk([], "empty-doc")
    assert chunks == [], "Empty elements should yield no chunks"


def test_resume_canonical_strategy_unknown_section():
    from app.chunking.strategies.resume_canonical import ResumeCanonicalStrategy
    from app.registry.models import IngestionConfig
    config = IngestionConfig(
        strategy="resume_canonical",
        max_tokens=512,
        min_tokens=10,
        overlap_tokens=0,
        semantic_split=False,
    )
    strategy = ResumeCanonicalStrategy(config)
    elements = [
        {"text": "Jane Smith", "page": 1},
        {"text": "Senior developer with 5 years of experience", "page": 1},
    ]
    chunks = strategy.chunk(elements, "test-doc-002")
    assert any(c.canonical_type == "misc" for c in chunks), \
        "Pre-header content without a section should be 'misc'"


# ---------------------------------------------------------------------------
# Registry tests (no LLM)
# ---------------------------------------------------------------------------

def test_resumelab_registry_loads():
    from app.registry.store import get_app
    config = get_app("resumelab")
    assert config is not None, "resumelab must be registered in registry.json"
    assert config.collection == "resume_memory"
    assert config.ingestion.strategy == "resume_canonical"
    assert config.retrieval.top_k == 12
    assert config.retrieval.hybrid is True
    assert config.retrieval.alpha == 0.7
    assert config.reranking.enabled is True
    assert config.reranking.top_k == 6


def test_resumelab_registry_backward_compat():
    from app.registry.store import get_app
    for app_name in ("doclens", "cvscan"):
        config = get_app(app_name)
        assert config is not None, f"{app_name} must still be registered (backward compat)"


def test_chunker_registry_includes_resume_canonical():
    from app.chunking.chunker import _REGISTRY
    assert "resume_canonical" in _REGISTRY, \
        "resume_canonical must be registered in chunker._REGISTRY"


# ---------------------------------------------------------------------------
# Integration tests: extract_resume with mocked LLM
# ---------------------------------------------------------------------------

def test_extract_resume_clean_text():
    """Happy path: clean resume text with mocked LLM."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_VALID_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=CLEAN_RESUME, doc_id="test-clean-001")

    assert result["doc_id"] == "test-clean-001"
    assert result["document_type"] == "resume"
    assert isinstance(result["skills"], list)
    assert isinstance(result["experience"], list)
    assert isinstance(result["education"], list)
    assert isinstance(result["certifications"], list)
    assert isinstance(result["keywords"], list)
    assert isinstance(result["raw_sections"], dict)
    assert "source_type" in result["metadata"]
    assert result["metadata"]["source_type"] == "text"
    assert 0.0 <= result["metadata"]["confidence"] <= 1.0


def test_extract_resume_malformed_first_attempt():
    """LLM returns markdown-fenced JSON on first attempt; should succeed after repair."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        # First response has fences; _repair_json should handle it
        mock_llm.generate.return_value = _MOCK_MALFORMED_FIRST
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=CLEAN_RESUME)

    assert result["document_type"] == "resume"
    assert isinstance(result["skills"], list)


def test_extract_resume_retry_on_missing_keys():
    """LLM returns invalid JSON on first attempt, valid on second (retry loop)."""
    call_count = [0]

    def side_effect(prompt, temperature=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return _MOCK_MISSING_KEY  # missing required keys
        return _MOCK_VALID_RESPONSE   # valid on retry

    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = side_effect
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=CLEAN_RESUME)

    assert call_count[0] == 2, f"Expected 2 LLM calls (1 fail + 1 retry), got {call_count[0]}"
    assert isinstance(result["experience"], list)


def test_extract_resume_all_retries_exhausted():
    """All retries fail: extract_resume must raise ValueError, not silently return."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "this is not json"
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        try:
            extract_resume(text=CLEAN_RESUME)
            raise AssertionError("Should have raised ValueError after exhausting retries")
        except ValueError as exc:
            assert "failed" in str(exc).lower() or "attempt" in str(exc).lower()


def test_extract_resume_messy_text():
    """Messy resume with duplicate sections should not crash."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_VALID_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=MESSY_RESUME)

    assert result is not None
    assert "metadata" in result


def test_extract_resume_missing_headings():
    """Resume with no standard section headings still produces a result."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_VALID_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=MISSING_HEADINGS_RESUME)

    assert isinstance(result["raw_sections"], dict)


def test_extract_resume_jd_like_input():
    """Non-resume input (JD text) should still extract with generic_profile type."""
    with patch("cortex.core.resume_extractor.get_llm") as mock_factory, \
         patch("cortex.core.resume_extractor._resolve_llm_config", return_value=None):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_VALID_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_extractor import extract_resume
        result = extract_resume(text=JD_LIKE_TEXT, extraction_type="generic_profile")

    assert result is not None
    assert result["metadata"]["source_type"] == "text"


def test_extract_resume_empty_text_raises():
    """Empty text must raise ValueError immediately, before LLM call."""
    from cortex.core.resume_extractor import extract_resume
    try:
        extract_resume(text="   ")
        raise AssertionError("Should have raised ValueError for empty text")
    except ValueError as exc:
        assert "text" in str(exc).lower() or "extractable" in str(exc).lower()


def test_extract_resume_no_input_raises():
    """Calling with no text and no file_path must raise ValueError."""
    from cortex.core.resume_extractor import extract_resume
    try:
        extract_resume()
        raise AssertionError("Should have raised ValueError for missing input")
    except ValueError as exc:
        assert "file_path" in str(exc).lower() or "text" in str(exc).lower()


def test_extract_resume_both_inputs_raises():
    """Providing both text and file_path must raise ValueError."""
    from cortex.core.resume_extractor import extract_resume
    try:
        extract_resume(text="some text", file_path="/some/path.pdf")
        raise AssertionError("Should have raised ValueError for both inputs")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# ExtractResponse schema validation
# ---------------------------------------------------------------------------

def test_extract_response_schema_valid():
    from cortex.schemas.resumelab import ExtractResponse, ExtractMetadata
    resp = ExtractResponse(
        doc_id="abc123",
        document_type="resume",
        skills=[{"name": "Python"}],
        projects=[],
        experience=[],
        education=[],
        certifications=[],
        keywords=["Python"],
        raw_sections={"skills": "Python"},
        metadata=ExtractMetadata(
            source_type="text",
            parsed_at="2026-05-01T00:00:00+00:00",
            confidence=0.9,
        ),
    )
    assert resp.doc_id == "abc123"
    assert resp.metadata.confidence == 0.9


def test_extract_response_confidence_bounds():
    from cortex.schemas.resumelab import ExtractMetadata
    from pydantic import ValidationError
    try:
        ExtractMetadata(source_type="text", parsed_at="2026-05-01T00:00:00+00:00", confidence=1.5)
        raise AssertionError("Should reject confidence > 1.0")
    except ValidationError:
        pass

    try:
        ExtractMetadata(source_type="text", parsed_at="2026-05-01T00:00:00+00:00", confidence=-0.1)
        raise AssertionError("Should reject confidence < 0.0")
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(run_llm_tests: bool = True) -> int:
    print("\n=== ResumeLab Extract Test Suite ===\n")

    print("-- Section Splitter --")
    _run("section splitter: clean resume", test_section_splitter_clean)
    _run("section splitter: alias normalization", test_section_splitter_alias_normalization)
    _run("section splitter: missing headings", test_section_splitter_missing_headings)
    _run("section splitter: empty input", test_section_splitter_empty_input)
    _run("section splitter: duplicate headings", test_section_splitter_duplicate_headings)

    print("\n-- JSON Repair --")
    _run("json repair: clean object", test_json_repair_clean_object)
    _run("json repair: strips markdown fence", test_json_repair_strips_markdown_fence)
    _run("json repair: extracts from prose", test_json_repair_extracts_from_prose)
    _run("json repair: raises on garbage", test_json_repair_raises_on_garbage)

    print("\n-- Schema Validator --")
    _run("schema validator: valid payload", test_validate_extract_payload_valid)
    _run("schema validator: missing key", test_validate_extract_payload_missing_key)
    _run("schema validator: wrong type", test_validate_extract_payload_wrong_type)

    print("\n-- ResumeCanonicalStrategy --")
    _run("canonical strategy: clean elements", test_resume_canonical_strategy_clean)
    _run("canonical strategy: empty elements", test_resume_canonical_strategy_empty_elements)
    _run("canonical strategy: unknown section → misc", test_resume_canonical_strategy_unknown_section)

    print("\n-- Registry --")
    _run("registry: resumelab loads", test_resumelab_registry_loads)
    _run("registry: backward compat (doclens/cvscan)", test_resumelab_registry_backward_compat)
    _run("registry: chunker includes resume_canonical", test_chunker_registry_includes_resume_canonical)

    print("\n-- extract_resume() with mocked LLM --")
    _run("extract: clean text (mocked)", test_extract_resume_clean_text, skip=not run_llm_tests)
    _run("extract: malformed first attempt (mocked)", test_extract_resume_malformed_first_attempt, skip=not run_llm_tests)
    _run("extract: retry on missing keys (mocked)", test_extract_resume_retry_on_missing_keys, skip=not run_llm_tests)
    _run("extract: all retries exhausted raises (mocked)", test_extract_resume_all_retries_exhausted, skip=not run_llm_tests)
    _run("extract: messy resume (mocked)", test_extract_resume_messy_text, skip=not run_llm_tests)
    _run("extract: missing headings (mocked)", test_extract_resume_missing_headings, skip=not run_llm_tests)
    _run("extract: JD-like input (mocked)", test_extract_resume_jd_like_input, skip=not run_llm_tests)
    _run("extract: empty text raises", test_extract_resume_empty_text_raises)
    _run("extract: no input raises", test_extract_resume_no_input_raises)
    _run("extract: both inputs raises", test_extract_resume_both_inputs_raises)

    print("\n-- ExtractResponse schema --")
    _run("ExtractResponse: valid shape", test_extract_response_schema_valid)
    _run("ExtractResponse: confidence bounds enforced", test_extract_response_confidence_bounds)

    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    errors = sum(1 for r in _results if r["status"] == "ERROR")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")
    total = len(_results)

    print(f"\n=== Results: {passed}/{total} passed | {failed} failed | {errors} errors | {skipped} skipped ===\n")

    if failed or errors:
        print("Failed/Errored tests:")
        for r in _results:
            if r["status"] in ("FAIL", "ERROR"):
                print(f"  [{r['status']}] {r['name']}: {r.get('error', '')}")

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResumeLab extract test suite")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip tests that patch the LLM (use if mock imports are unavailable)",
    )
    args = parser.parse_args()
    sys.exit(main(run_llm_tests=not args.no_llm))
