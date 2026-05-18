"""
ResumeLab /analyze/match test suite.

Run:
    python -m scripts.test_resumelab_match
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable, Dict, List
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Harness
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BACKEND_PROFILE = {
    "doc_id": "doc-backend",
    "skills": [
        {"normalized_name": "Python", "category": "programming", "proficiency": "expert", "aliases": ["Python"], "sources": ["doc-001"]},
        {"normalized_name": "AWS", "category": "cloud", "proficiency": "advanced", "aliases": ["AWS", "Amazon Web Services"], "sources": ["doc-001"]},
        {"normalized_name": "PostgreSQL", "category": "database", "proficiency": "proficient", "aliases": ["PostgreSQL", "Postgres"], "sources": ["doc-001"]},
        {"normalized_name": "Kubernetes", "category": "devops", "proficiency": "proficient", "aliases": ["Kubernetes", "k8s"], "sources": ["doc-001"]},
        {"normalized_name": "Go", "category": "programming", "proficiency": "intermediate", "aliases": ["Go", "Golang"], "sources": ["doc-001"]},
        {"normalized_name": "React", "category": "frontend", "proficiency": "beginner", "aliases": ["React"], "sources": ["doc-001"]},
    ],
    "experience": [
        {"canonical_key": "google_swe", "company": "Google", "title": "Senior Software Engineer",
         "date_range": "2020-Present", "location": "Mountain View, CA",
         "bullets": ["Led Kubernetes-based microservices migration", "Built REST APIs serving 10M RPD"],
         "sources": ["doc-001"]},
        {"canonical_key": "meta_eng", "company": "Meta", "title": "Software Engineer",
         "date_range": "2017-2020", "location": "Menlo Park, CA",
         "bullets": ["Kafka pipeline processing 50M events/day", "AWS Lambda ETL automation"],
         "sources": ["doc-001"]},
    ],
    "projects": [
        {"canonical_key": "dataflow", "normalized_name": "DataFlow",
         "description": "Distributed ETL framework", "technologies": ["Python", "Airflow", "Spark"],
         "url": "github.com/janesmith/dataflow", "sources": ["doc-001"]},
    ],
    "certifications": [
        {"canonical_key": "aws_csa", "normalized_name": "AWS Certified Solutions Architect",
         "aliases": ["AWS Certified Solutions Architect"], "sources": ["doc-001"]},
    ],
    "keywords": ["Python", "AWS", "microservices", "Kubernetes"],
    "source_documents": ["doc-001"],
    "canonical_ids": [],
    "merged_at": "2026-05-02T00:00:00+00:00",
}

# Base resume only shows Python and Go — AWS, PostgreSQL, Kubernetes are OMITTED
BASE_RESUME_INCOMPLETE = {
    "doc_id": "doc-base",
    "skills": [
        {"name": "Python"},
        {"name": "Go"},
    ],
    "experience": [
        {"company": "Google", "title": "Senior Software Engineer", "date_range": "2020-Present", "bullets": ["Built APIs"]},
    ],
    "projects": [],
    "certifications": [],
    "keywords": ["Python", "Go"],
}

FRONTEND_JD = """
Senior Frontend Engineer

We are seeking a Senior Frontend Engineer to join our product team.

Requirements:
- 5+ years of React experience
- TypeScript proficiency
- Experience with state management (Redux, Zustand, or similar)
- CSS-in-JS (styled-components, Tailwind)
- Performance optimization and Core Web Vitals
- Accessibility (WCAG 2.1 AA)

Nice to have:
- Next.js experience
- GraphQL
- Jest and testing-library

The ideal candidate will own our design system and lead performance initiatives.
"""

BACKEND_JD = """
Staff Backend Engineer — Platform Team

Requirements:
- 7+ years backend engineering
- Python or Go (we use both)
- Distributed systems design
- AWS (EC2, ECS, RDS, Lambda)
- PostgreSQL at scale (query optimization, indexing)
- Kubernetes, Docker
- REST API design, OpenAPI

Responsibilities:
- Design and own high-throughput data pipelines
- Lead API platform evolution
- Mentor 3-4 engineers
- Partner with ML team on infrastructure
"""

MALFORMED_JD = "   \n\n   "  # whitespace only

# Canonical mock LLM response for a match analysis
MOCK_MATCH_RESPONSE = json.dumps({
    "match_score": 78,
    "required_keywords": ["Python", "Go", "AWS", "PostgreSQL", "Kubernetes", "Docker", "REST API"],
    "missing_keywords": ["Docker"],
    "existing_but_missing_from_resume": ["AWS", "PostgreSQL", "Kubernetes"],
    "irrelevant_content": ["React (low relevance for backend role)"],
    "recommended_additions": ["AWS Certified Solutions Architect", "Kubernetes experience from Google role"],
    "recommended_removals": ["React (beginner level, not relevant)"],
    "section_rewrites": {
        "summary": "Staff-level backend engineer with 7+ years in Python and Go, specializing in distributed systems and AWS infrastructure.",
        "skills": ["Python", "Go", "AWS", "PostgreSQL", "Kubernetes", "REST API"],
        "projects": [{"name": "DataFlow", "description": "Distributed ETL framework", "technologies": ["Python", "Airflow", "Spark"]}]
    },
    "ats_keyword_clusters": {
        "languages": ["Python", "Go"],
        "cloud": ["AWS"],
        "databases": ["PostgreSQL"],
        "devops": ["Kubernetes", "Docker"]
    },
    "role_seniority": "senior",
    "domain_fit": "Strong backend and distributed systems alignment; candidate has directly relevant AWS and Kubernetes experience."
})


# ---------------------------------------------------------------------------
# Unit tests: keyword helpers
# ---------------------------------------------------------------------------

def test_extract_profile_skills():
    from cortex.core.resume_optimizer import _extract_profile_skills
    skills = _extract_profile_skills(BACKEND_PROFILE)
    assert "python" in skills
    assert "aws" in skills
    assert "amazon web services" in skills  # alias
    assert "kubernetes" in skills
    assert "k8s" in skills  # alias
    # Technologies from projects
    assert "airflow" in skills
    assert "spark" in skills


def test_extract_resume_skills():
    from cortex.core.resume_optimizer import _extract_resume_skills
    skills = _extract_resume_skills(BASE_RESUME_INCOMPLETE)
    assert "python" in skills
    assert "go" in skills
    assert "aws" not in skills     # omitted
    assert "postgresql" not in skills  # omitted


def test_extract_resume_skills_none():
    from cortex.core.resume_optimizer import _extract_resume_skills
    skills = _extract_resume_skills(None)
    assert skills == set()


def test_compute_keyword_sets_omission_detection():
    """Core A/B distinction: profile_skills - resume_skills = omitted skills."""
    from cortex.core.resume_optimizer import _compute_keyword_sets
    profile_s, resume_s, omitted = _compute_keyword_sets(BACKEND_PROFILE, BASE_RESUME_INCOMPLETE)

    # Profile has these; resume doesn't
    assert "aws" in omitted, "AWS in profile but not in base resume → must be in omitted"
    assert "postgresql" in omitted
    assert "kubernetes" in omitted

    # Resume has these; they should NOT be in omitted
    assert "python" not in omitted
    assert "go" not in omitted


def test_compute_keyword_sets_no_resume():
    """Without a base_resume, all profile skills are 'omitted' (everything is an opportunity)."""
    from cortex.core.resume_optimizer import _compute_keyword_sets
    profile_s, resume_s, omitted = _compute_keyword_sets(BACKEND_PROFILE, None)
    assert resume_s == set()
    assert omitted == profile_s


def test_compute_keyword_sets_empty_profile():
    from cortex.core.resume_optimizer import _compute_keyword_sets
    profile_s, resume_s, omitted = _compute_keyword_sets({}, BASE_RESUME_INCOMPLETE)
    assert profile_s == set()
    assert omitted == set()


# ---------------------------------------------------------------------------
# Unit tests: profile formatter
# ---------------------------------------------------------------------------

def test_format_profile_context_has_sections():
    from cortex.core.resume_optimizer import (
        _format_profile_context, _compute_keyword_sets
    )
    profile_s, resume_s, omitted = _compute_keyword_sets(BACKEND_PROFILE, BASE_RESUME_INCOMPLETE)
    ctx = _format_profile_context(BACKEND_PROFILE, profile_s, resume_s, omitted)

    assert "CANONICAL PROFILE" in ctx
    assert "PRE-COMPUTED KEYWORD ANALYSIS" in ctx
    assert "SKILLS:" in ctx
    assert "EXPERIENCE:" in ctx
    assert "PROJECTS:" in ctx
    # Should mention omitted skills
    assert "aws" in ctx.lower() or "AWS" in ctx


def test_format_base_resume_not_provided():
    from cortex.core.resume_optimizer import _format_base_resume
    text = _format_base_resume(None)
    assert "Not provided" in text


def test_format_base_resume_with_skills():
    from cortex.core.resume_optimizer import _format_base_resume
    text = _format_base_resume(BASE_RESUME_INCOMPLETE)
    assert "Python" in text or "python" in text.lower()


# ---------------------------------------------------------------------------
# Integration tests: analyze_match (mocked LLM)
# ---------------------------------------------------------------------------

def test_analyze_match_happy_path():
    """Backend JD vs backend profile — clean success case."""
    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MOCK_MATCH_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        llm_config = LLMConfig(provider="ollama_local", model="llama3")
        schema = {"type": "object", "required": ["match_score", "required_keywords", "missing_keywords",
                  "existing_but_missing_from_resume", "irrelevant_content", "recommended_additions",
                  "recommended_removals", "section_rewrites", "ats_keyword_clusters",
                  "role_seniority", "domain_fit"]}
        result = analyze_match(
            job_description=BACKEND_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=BASE_RESUME_INCOMPLETE,
            llm_config=llm_config,
            system_prompt="You are a career intelligence engine.",
            schema=schema,
        )

    assert isinstance(result["match_score"], (int, float))
    assert 0 <= result["match_score"] <= 100
    assert isinstance(result["required_keywords"], list)
    assert isinstance(result["missing_keywords"], list)
    assert isinstance(result["existing_but_missing_from_resume"], list)
    assert isinstance(result["section_rewrites"], dict)
    assert isinstance(result["ats_keyword_clusters"], dict)
    assert result["role_seniority"] in (
        "junior", "mid", "senior", "lead", "principal", "executive", "staff"
    ) or isinstance(result["role_seniority"], str)


def test_analyze_match_identifies_omitted_skills():
    """The LLM output's existing_but_missing_from_resume must reflect our A/B pre-computation."""
    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MOCK_MATCH_RESPONSE
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        result = analyze_match(
            job_description=BACKEND_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=BASE_RESUME_INCOMPLETE,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="You are a career intelligence engine.",
            schema={},
        )

    # The mock response includes these — verify the pipeline passes them through
    assert "AWS" in result["existing_but_missing_from_resume"] or \
           any("AWS" in k for k in result["existing_but_missing_from_resume"])


def test_analyze_match_frontend_jd_vs_backend_profile():
    """Frontend JD vs backend-heavy profile should surface React/TypeScript gaps."""
    frontend_mock = json.dumps({
        "match_score": 35,
        "required_keywords": ["React", "TypeScript", "CSS", "Next.js", "Redux"],
        "missing_keywords": ["TypeScript", "Next.js", "Redux", "CSS-in-JS"],
        "existing_but_missing_from_resume": ["React"],
        "irrelevant_content": ["AWS Lambda", "Kafka pipeline", "PostgreSQL optimization"],
        "recommended_additions": ["React (already in profile — add to resume)"],
        "recommended_removals": ["Kafka pipeline — not relevant for frontend role"],
        "section_rewrites": {"summary": "Frontend engineer with React skills.", "skills": ["React"], "projects": []},
        "ats_keyword_clusters": {"frontend": ["React"], "missing": ["TypeScript", "Next.js"]},
        "role_seniority": "senior",
        "domain_fit": "Candidate is primarily a backend engineer; limited frontend depth beyond React."
    })

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = frontend_mock
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        result = analyze_match(
            job_description=FRONTEND_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=None,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="You are a career intelligence engine.",
            schema={},
        )

    assert result["match_score"] < 60, "Backend profile should score low vs frontend JD"
    assert len(result["missing_keywords"]) > 0, "TypeScript/Next.js should be flagged as missing"
    assert len(result["irrelevant_content"]) > 0, "Backend content should be flagged irrelevant"


def test_analyze_match_empty_jd_raises():
    from cortex.core.resume_optimizer import analyze_match
    from app.context import LLMConfig
    try:
        analyze_match(
            job_description="",
            canonical_profile=BACKEND_PROFILE,
            base_resume=None,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )
        raise AssertionError("Should have raised ValueError for empty JD")
    except ValueError as exc:
        assert "job_description" in str(exc).lower() or "empty" in str(exc).lower()


def test_analyze_match_malformed_jd_raises():
    from cortex.core.resume_optimizer import analyze_match
    from app.context import LLMConfig
    try:
        analyze_match(
            job_description=MALFORMED_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=None,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )
        raise AssertionError("Should have raised ValueError for whitespace-only JD")
    except ValueError:
        pass


def test_analyze_match_retry_on_bad_json():
    """LLM returns garbage first, valid JSON second — retry loop catches it."""
    call_count = [0]

    def side_effect(prompt, temperature=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return "this is not json at all"
        return MOCK_MATCH_RESPONSE

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = side_effect
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        result = analyze_match(
            job_description=BACKEND_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=None,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )

    assert call_count[0] == 2, f"Expected 2 LLM calls, got {call_count[0]}"
    assert isinstance(result["match_score"], (int, float))


def test_analyze_match_all_retries_exhausted_raises():
    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not json"
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        try:
            analyze_match(
                job_description=BACKEND_JD,
                canonical_profile=BACKEND_PROFILE,
                base_resume=None,
                llm_config=LLMConfig(provider="ollama_local", model="llama3"),
                system_prompt="",
                schema={},
                max_retries=1,
            )
            raise AssertionError("Should have raised ValueError after exhausting retries")
        except ValueError as exc:
            assert "failed" in str(exc).lower() or "attempt" in str(exc).lower()


def test_analyze_match_match_score_clamped():
    """match_score must be clamped to 0-100 even if LLM returns out-of-range value."""
    mock_response = json.dumps({
        "match_score": 150,  # out of range
        "required_keywords": [],
        "missing_keywords": [],
        "existing_but_missing_from_resume": [],
        "irrelevant_content": [],
        "recommended_additions": [],
        "recommended_removals": [],
        "section_rewrites": {},
        "ats_keyword_clusters": {},
        "role_seniority": "senior",
        "domain_fit": "good fit",
    })

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = mock_response
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import analyze_match
        from app.context import LLMConfig
        result = analyze_match(
            job_description=BACKEND_JD,
            canonical_profile=BACKEND_PROFILE,
            base_resume=None,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )

    assert result["match_score"] <= 100.0, f"match_score must be clamped: {result['match_score']}"


def test_analyze_match_omitted_aws_history():
    """Classic 'AWS history omitted' scenario: candidate has AWS but it's not in base resume."""
    from cortex.core.resume_optimizer import _compute_keyword_sets
    profile_s, resume_s, omitted = _compute_keyword_sets(BACKEND_PROFILE, BASE_RESUME_INCOMPLETE)

    # AWS is in profile via aliases, not in base resume
    assert "aws" in omitted, (
        f"AWS (and aliases) must appear in omitted. "
        f"profile_skills={sorted(profile_s)[:10]}, resume_skills={sorted(resume_s)}"
    )
    assert "amazon web services" in omitted  # alias also tracked


def test_analyze_match_perfect_case():
    """Profile with every JD keyword in base_resume → no omissions."""
    full_resume = {
        "skills": [
            {"name": "Python"}, {"name": "Go"}, {"name": "AWS"},
            {"name": "PostgreSQL"}, {"name": "Kubernetes"}, {"name": "React"},
        ],
        "keywords": ["AWS", "PostgreSQL", "Kubernetes"],
    }
    from cortex.core.resume_optimizer import _compute_keyword_sets
    _, _, omitted = _compute_keyword_sets(BACKEND_PROFILE, full_resume)
    # Some aliases might still be "omitted" but core skills should be covered
    core_omitted = {k for k in omitted if k in {"python", "go", "aws", "postgresql", "kubernetes"}}
    assert len(core_omitted) == 0, f"Core skills should not be in omitted: {core_omitted}"


# ---------------------------------------------------------------------------
# Registry + route tests
# ---------------------------------------------------------------------------

def test_match_task_in_registry():
    from app.registry.store import get_app
    config = get_app("resumelab")
    assert "match" in config.tasks
    task = config.tasks["match"]
    assert task.output_schema is not None
    required = task.output_schema.get("required", [])
    assert "match_score" in required
    assert "existing_but_missing_from_resume" in required
    assert "section_rewrites" in required


def test_match_route_registered():
    from cortex.api.main import app
    routes = [r.path for r in app.routes]
    assert "/analyze/match" in routes, "/analyze/match must be registered"


def test_match_request_schema():
    from cortex.schemas.resumelab import MatchRequest
    req = MatchRequest(
        app_name="resumelab",
        user_id="u1",
        job_description="We need a Python engineer",
        canonical_profile={"skills": [{"name": "Python"}]},
    )
    assert req.base_resume is None


def test_match_response_schema_valid():
    from cortex.schemas.resumelab import MatchResponse
    resp = MatchResponse(
        match_score=75.0,
        required_keywords=["Python", "AWS"],
        missing_keywords=[],
        existing_but_missing_from_resume=["AWS"],
        irrelevant_content=["Flutter"],
        recommended_additions=["Add AWS to skills section"],
        recommended_removals=[],
        section_rewrites={"summary": "Python engineer with AWS background."},
        ats_keyword_clusters={"cloud": ["AWS"]},
        role_seniority="senior",
        domain_fit="Strong backend alignment.",
    )
    assert resp.match_score == 75.0
    assert "AWS" in resp.existing_but_missing_from_resume


def test_match_response_score_bounds():
    from cortex.schemas.resumelab import MatchResponse
    from pydantic import ValidationError
    try:
        MatchResponse(
            match_score=105.0,  # out of range
            required_keywords=[], missing_keywords=[],
            existing_but_missing_from_resume=[], irrelevant_content=[],
            recommended_additions=[], recommended_removals=[],
            section_rewrites={}, ats_keyword_clusters={},
            role_seniority="mid", domain_fit="ok",
        )
        raise AssertionError("Should reject score > 100")
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n=== ResumeLab Match Test Suite ===\n")

    print("-- Keyword Helpers --")
    _run("extract profile skills (with aliases)", test_extract_profile_skills)
    _run("extract resume skills", test_extract_resume_skills)
    _run("extract resume skills (None)", test_extract_resume_skills_none)
    _run("compute keyword sets: omission detection", test_compute_keyword_sets_omission_detection)
    _run("compute keyword sets: no resume", test_compute_keyword_sets_no_resume)
    _run("compute keyword sets: empty profile", test_compute_keyword_sets_empty_profile)

    print("\n-- Profile Formatter --")
    _run("format profile context has all sections", test_format_profile_context_has_sections)
    _run("format base resume: not provided", test_format_base_resume_not_provided)
    _run("format base resume: with skills", test_format_base_resume_with_skills)

    print("\n-- analyze_match() --")
    _run("happy path: backend JD vs backend profile", test_analyze_match_happy_path)
    _run("identifies omitted skills (B category)", test_analyze_match_identifies_omitted_skills)
    _run("frontend JD vs backend profile (low score)", test_analyze_match_frontend_jd_vs_backend_profile)
    _run("empty JD raises ValueError", test_analyze_match_empty_jd_raises)
    _run("malformed JD raises ValueError", test_analyze_match_malformed_jd_raises)
    _run("retry on bad JSON", test_analyze_match_retry_on_bad_json)
    _run("all retries exhausted raises", test_analyze_match_all_retries_exhausted_raises)
    _run("match_score clamped to 0-100", test_analyze_match_match_score_clamped)
    _run("omitted AWS history detected", test_analyze_match_omitted_aws_history)
    _run("perfect match: no omissions", test_analyze_match_perfect_case)

    print("\n-- Registry + Routes --")
    _run("match task in registry", test_match_task_in_registry)
    _run("/analyze/match route registered", test_match_route_registered)
    _run("MatchRequest schema valid", test_match_request_schema)
    _run("MatchResponse schema valid", test_match_response_schema_valid)
    _run("MatchResponse score bounds enforced", test_match_response_score_bounds)

    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    errors = sum(1 for r in _results if r["status"] == "ERROR")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")
    total = len(_results)

    print(f"\n=== Results: {passed}/{total} passed | {failed} failed | {errors} errors | {skipped} skipped ===\n")
    if failed or errors:
        for r in _results:
            if r["status"] in ("FAIL", "ERROR"):
                print(f"  [{r['status']}] {r['name']}: {r.get('error', '')}")

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
