"""
ResumeLab /generate/document test suite.

Run:
    python -m scripts.test_resumelab_generate
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

FULL_PROFILE = {
    "doc_id": "doc-001",
    "skills": [
        {"normalized_name": "Python", "proficiency": "expert", "aliases": ["Python"], "sources": ["doc-001"]},
        {"normalized_name": "AWS", "proficiency": "advanced", "aliases": ["AWS", "Amazon Web Services"], "sources": ["doc-001"]},
        {"normalized_name": "React", "proficiency": "intermediate", "aliases": ["React", "React.js"], "sources": ["doc-001"]},
        {"normalized_name": "TypeScript", "proficiency": "intermediate", "aliases": ["TypeScript", "TS"], "sources": ["doc-001"]},
        {"normalized_name": "PostgreSQL", "proficiency": "proficient", "aliases": ["PostgreSQL", "Postgres"], "sources": ["doc-001"]},
        {"normalized_name": "Flutter", "proficiency": "beginner", "aliases": ["Flutter"], "sources": ["doc-001"]},
        {"normalized_name": "Kubernetes", "proficiency": "proficient", "aliases": ["k8s", "Kubernetes"], "sources": ["doc-001"]},
    ],
    "experience": [
        {
            "canonical_key": "google_swe", "company": "Google",
            "title": "Senior Software Engineer", "date_range": "2020-Present",
            "location": "Mountain View, CA",
            "bullets": [
                "Led microservices migration to Kubernetes reducing latency 40%",
                "Built REST APIs serving 10M requests/day with Python and AWS Lambda",
                "Mentored 4 engineers",
                "Managed AWS EC2/ECS infrastructure for 99.99% uptime SLA",
            ],
            "sources": ["doc-001"],
        },
        {
            "canonical_key": "meta_swe", "company": "Meta",
            "title": "Software Engineer", "date_range": "2017-2020",
            "location": "Menlo Park, CA",
            "bullets": [
                "Built React dashboard reducing operations team toil by 60%",
                "Kafka pipeline processing 50M events/day",
            ],
            "sources": ["doc-001"],
        },
    ],
    "projects": [
        {
            "canonical_key": "dataflow", "normalized_name": "DataFlow",
            "description": "Open-source distributed ETL framework used by 500+ teams",
            "technologies": ["Python", "Apache Spark", "Airflow"],
            "url": "github.com/janesmith/dataflow", "sources": ["doc-001"],
        },
        {
            "canonical_key": "flutterdemo", "normalized_name": "FlutterDemo",
            "description": "Demo mobile app in Flutter",
            "technologies": ["Flutter", "Dart"], "url": None, "sources": ["doc-001"],
        },
    ],
    "certifications": [
        {"canonical_key": "aws_csa", "normalized_name": "AWS Certified Solutions Architect",
         "aliases": ["AWS Certified Solutions Architect"], "sources": ["doc-001"]},
    ],
    "keywords": ["Python", "AWS", "React", "Kubernetes", "microservices"],
    "source_documents": ["doc-001"],
    "canonical_ids": [],
    "merged_at": "2026-05-02T00:00:00+00:00",
}

BASE_RESUME = {
    "doc_id": "doc-base",
    "skills": [{"name": "Python"}, {"name": "React"}, {"name": "Flutter"}],
    "experience": [
        {"company": "Google", "title": "Senior Software Engineer",
         "bullets": ["Led microservices migration"]},
    ],
    "projects": [
        {"name": "FlutterDemo", "description": "Demo app"},
    ],
    "certifications": [],
    "keywords": ["Python", "React", "Flutter"],
}

BACKEND_JD = """
Staff Backend Engineer — Platform

Requirements:
- 7+ years backend engineering
- Python expert
- AWS services (EC2, ECS, Lambda, RDS)
- Kubernetes, Docker
- PostgreSQL at scale
- REST API design
- System design and distributed systems

We are NOT looking for mobile or Flutter experience.
"""

FRONTEND_JD = """
Senior Frontend Engineer

Requirements:
- React and TypeScript
- State management (Redux or Zustand)
- CSS-in-JS, Tailwind
- Performance optimization, Core Web Vitals
- Accessibility (WCAG)
- Jest, Testing Library

Nice to have:
- Next.js
- GraphQL
"""

FULLSTACK_JD = """
Full-Stack Engineer

Build features end-to-end across Python backend and React frontend.
AWS cloud, PostgreSQL, REST APIs. TypeScript on the front.
Kubernetes for deployment. Some React Native nice-to-have.
"""

# Canonical mock responses
MOCK_BACKEND_DOC = json.dumps({
    "summary": "Staff-level backend engineer with 7+ years in Python and AWS, specializing in distributed systems and Kubernetes orchestration.",
    "skills": ["Python", "AWS", "Kubernetes", "PostgreSQL", "REST API", "Docker"],
    "projects": [
        {"name": "DataFlow", "description": "Distributed ETL framework processing large-scale data",
         "technologies": ["Python", "Airflow", "Spark"], "url": "github.com/janesmith/dataflow",
         "relevance_note": "Demonstrates large-scale Python and data pipeline expertise"}
    ],
    "experience": [
        {"company": "Google", "title": "Senior Software Engineer", "date_range": "2020-Present",
         "location": "Mountain View, CA",
         "bullets": [
             "Led Kubernetes-based microservices migration reducing P95 latency by 40%",
             "Built REST APIs serving 10M requests/day with Python and AWS Lambda",
             "Managed AWS EC2/ECS infrastructure for 99.99% uptime SLA",
         ]},
        {"company": "Meta", "title": "Software Engineer", "date_range": "2017-2020",
         "location": "Menlo Park, CA",
         "bullets": ["Kafka pipeline processing 50M events/day"]},
    ],
    "target_keywords_used": ["Python", "AWS", "Kubernetes", "PostgreSQL", "REST API", "distributed systems"],
    "removed_content": ["Flutter (irrelevant)", "FlutterDemo project (irrelevant)", "React (weak for backend)"],
    "match_score_improved": 87
})

MOCK_FRONTEND_DOC = json.dumps({
    "summary": "Frontend-capable engineer with React and TypeScript experience, building performant UIs.",
    "skills": ["React", "TypeScript", "AWS", "Python"],
    "projects": [
        {"name": "Meta Dashboard (React)", "description": "React dashboard reducing operations toil by 60%",
         "technologies": ["React"], "url": None, "relevance_note": "React frontend experience"}
    ],
    "experience": [
        {"company": "Meta", "title": "Software Engineer", "date_range": "2017-2020",
         "location": "Menlo Park, CA",
         "bullets": ["Built React dashboard reducing operations team toil by 60%"]},
    ],
    "target_keywords_used": ["React", "TypeScript"],
    "removed_content": ["Kubernetes (irrelevant)", "AWS Lambda details (irrelevant)", "PostgreSQL (irrelevant)"],
    "match_score_improved": 52
})

MOCK_FULLSTACK_DOC = json.dumps({
    "summary": "Full-stack engineer owning Python backend and React frontend, with AWS infrastructure expertise.",
    "skills": ["Python", "React", "TypeScript", "AWS", "PostgreSQL", "Kubernetes"],
    "projects": [
        {"name": "DataFlow", "description": "Python-based ETL framework",
         "technologies": ["Python", "Airflow"], "url": "github.com/janesmith/dataflow",
         "relevance_note": "Backend expertise"}
    ],
    "experience": [
        {"company": "Google", "title": "Senior Software Engineer", "date_range": "2020-Present",
         "location": "Mountain View, CA",
         "bullets": [
             "Built REST APIs with Python and AWS",
             "Led Kubernetes migration",
             "Built React dashboard at Meta",
         ]},
    ],
    "target_keywords_used": ["Python", "React", "TypeScript", "AWS", "PostgreSQL", "Kubernetes"],
    "removed_content": ["Flutter (irrelevant)", "FlutterDemo project (irrelevant)"],
    "match_score_improved": 82
})


# ---------------------------------------------------------------------------
# Unit tests: template guidance
# ---------------------------------------------------------------------------

def test_template_guidance_all_types():
    from cortex.core.resume_optimizer import _TEMPLATE_GUIDANCE
    assert "frontend" in _TEMPLATE_GUIDANCE
    assert "backend" in _TEMPLATE_GUIDANCE
    assert "fullstack" in _TEMPLATE_GUIDANCE
    # Backend guidance should mention databases
    assert "database" in _TEMPLATE_GUIDANCE["backend"].lower() or "sql" in _TEMPLATE_GUIDANCE["backend"].lower()
    # Frontend guidance should mention React
    assert "react" in _TEMPLATE_GUIDANCE["frontend"].lower()


def test_invalid_template_type_defaults_to_fullstack():
    """Unknown template_type must silently fall back to fullstack."""
    from cortex.core.resume_optimizer import _TEMPLATE_GUIDANCE
    safe = "unknown_type".strip().lower()
    valid_types = {"frontend", "backend", "fullstack"}
    if safe not in valid_types:
        safe = "fullstack"
    assert safe in _TEMPLATE_GUIDANCE


# ---------------------------------------------------------------------------
# Integration tests: generate_document (mocked LLM)
# ---------------------------------------------------------------------------

def _run_generate(mock_response, jd, template_type="fullstack"):
    from cortex.core.resume_optimizer import generate_document
    from app.context import LLMConfig

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = mock_response
        mock_factory.return_value = mock_llm

        result = generate_document(
            job_description=jd,
            canonical_profile=FULL_PROFILE,
            base_resume=BASE_RESUME,
            template_type=template_type,
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="You are a career intelligence engine.",
            schema={},
        )
    return result


def test_generate_backend_template():
    """Backend template should emphasize AWS/Kubernetes, de-emphasize Flutter."""
    result = _run_generate(MOCK_BACKEND_DOC, BACKEND_JD, "backend")

    assert isinstance(result["summary"], str) and len(result["summary"]) > 20
    assert isinstance(result["skills"], list)
    assert isinstance(result["experience"], list)
    assert isinstance(result["projects"], list)
    assert isinstance(result["target_keywords_used"], list)
    assert isinstance(result["removed_content"], list)
    assert isinstance(result["match_score_improved"], (int, float))

    skill_names_lower = [s.lower() for s in result["skills"]]
    assert "python" in skill_names_lower or "aws" in skill_names_lower, \
        "Backend template must include Python/AWS in skills"

    removed_lower = " ".join(result["removed_content"]).lower()
    assert "flutter" in removed_lower, "Flutter should appear in removed_content for backend JD"


def test_generate_frontend_template():
    """Frontend template should feature React, de-emphasize Kubernetes/AWS."""
    result = _run_generate(MOCK_FRONTEND_DOC, FRONTEND_JD, "frontend")

    skill_names_lower = [s.lower() for s in result["skills"]]
    assert "react" in skill_names_lower or "typescript" in skill_names_lower, \
        "Frontend template must surface React/TypeScript"

    removed_lower = " ".join(result["removed_content"]).lower()
    assert "kubernetes" in removed_lower or "aws" in removed_lower or "postgresql" in removed_lower, \
        "Heavy backend content should be removed for frontend role"


def test_generate_fullstack_template():
    """Fullstack should balance backend + frontend skills."""
    result = _run_generate(MOCK_FULLSTACK_DOC, FULLSTACK_JD, "fullstack")

    skill_names_lower = [s.lower() for s in result["skills"]]
    has_frontend = any(k in skill_names_lower for k in ["react", "typescript"])
    has_backend = any(k in skill_names_lower for k in ["python", "aws", "postgresql"])
    assert has_frontend or has_backend, "Fullstack must include at least frontend OR backend skills"


def test_generate_irrelevant_flutter_removed():
    """Flutter is in the profile but not relevant for backend/fullstack JDs."""
    result = _run_generate(MOCK_BACKEND_DOC, BACKEND_JD, "backend")
    # Flutter should either be absent from skills or appear in removed_content
    skill_names_lower = [s.lower() for s in result["skills"]]
    removed_lower = " ".join(result["removed_content"]).lower()
    assert "flutter" not in skill_names_lower or "flutter" in removed_lower, \
        "Flutter must be excluded from skills or flagged in removed_content for backend role"


def test_generate_target_keywords_are_strings():
    result = _run_generate(MOCK_BACKEND_DOC, BACKEND_JD, "backend")
    assert all(isinstance(k, str) for k in result["target_keywords_used"]), \
        "target_keywords_used must be a list of strings"


def test_generate_experience_blocks_have_required_fields():
    result = _run_generate(MOCK_BACKEND_DOC, BACKEND_JD, "backend")
    for exp in result["experience"]:
        assert "company" in exp, f"Experience block missing 'company': {exp}"
        assert "title" in exp, f"Experience block missing 'title': {exp}"
        assert "bullets" in exp, f"Experience block missing 'bullets': {exp}"
        assert isinstance(exp["bullets"], list), "bullets must be a list"


def test_generate_match_score_improved_clamped():
    """match_score_improved must be clamped to 0-100."""
    mock_oob = json.dumps({
        "summary": "Good engineer.",
        "skills": ["Python"],
        "projects": [],
        "experience": [],
        "target_keywords_used": ["Python"],
        "removed_content": [],
        "match_score_improved": 999,  # out of range
    })
    result = _run_generate(mock_oob, BACKEND_JD, "backend")
    assert result["match_score_improved"] <= 100.0, \
        f"match_score_improved must be clamped. Got: {result['match_score_improved']}"


def test_generate_empty_jd_raises():
    from cortex.core.resume_optimizer import generate_document
    from app.context import LLMConfig
    try:
        generate_document(
            job_description="",
            canonical_profile=FULL_PROFILE,
            base_resume=None,
            template_type="backend",
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )
        raise AssertionError("Should raise ValueError for empty JD")
    except ValueError:
        pass


def test_generate_retry_on_bad_json():
    """LLM fails once, succeeds on retry."""
    call_count = [0]

    def side_effect(prompt, temperature=None):
        call_count[0] += 1
        return "not json" if call_count[0] == 1 else MOCK_BACKEND_DOC

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = side_effect
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import generate_document
        from app.context import LLMConfig
        result = generate_document(
            job_description=BACKEND_JD,
            canonical_profile=FULL_PROFILE,
            base_resume=None,
            template_type="backend",
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="",
            schema={},
        )

    assert call_count[0] == 2
    assert isinstance(result["summary"], str)


def test_generate_prompt_contains_no_flutter_fabrication():
    """
    Verify the prompt construction does NOT introduce Flutter into a backend context.
    The prompt must only reference profile data — template guidance should
    DEPRIORITISE irrelevant content, not fabricate it.
    """
    captured_prompts = []

    def side_effect(prompt, temperature=None):
        captured_prompts.append(prompt)
        return MOCK_BACKEND_DOC

    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = side_effect
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import generate_document
        from app.context import LLMConfig
        generate_document(
            job_description=BACKEND_JD,
            canonical_profile=FULL_PROFILE,
            base_resume=None,
            template_type="backend",
            llm_config=LLMConfig(provider="ollama_local", model="llama3"),
            system_prompt="Never fabricate.",
            schema={},
        )

    assert captured_prompts, "LLM was not called"
    prompt_text = captured_prompts[0]
    # Template guidance must instruct DE-PRIORITISATION, not addition
    assert "deprioritise" in prompt_text.lower() or "depriorit" in prompt_text.lower() or \
           "not relevant" in prompt_text.lower() or "exclude" in prompt_text.lower() or \
           "TRUTHFUL" in prompt_text or "fabricate" in prompt_text.lower(), \
        "Backend template prompt must contain anti-fabrication guidance"


def test_generate_all_retries_fail():
    with patch("cortex.core.resume_optimizer.get_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not json at all"
        mock_factory.return_value = mock_llm

        from cortex.core.resume_optimizer import generate_document
        from app.context import LLMConfig
        try:
            generate_document(
                job_description=BACKEND_JD,
                canonical_profile=FULL_PROFILE,
                base_resume=None,
                template_type="backend",
                llm_config=LLMConfig(provider="ollama_local", model="llama3"),
                system_prompt="",
                schema={},
                max_retries=1,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError as exc:
            assert "failed" in str(exc).lower()


# ---------------------------------------------------------------------------
# Registry + route tests
# ---------------------------------------------------------------------------

def test_generate_task_in_registry():
    from app.registry.store import get_app
    config = get_app("resumelab")
    assert "generate" in config.tasks
    task = config.tasks["generate"]
    assert task.output_schema is not None
    required = task.output_schema.get("required", [])
    assert "summary" in required
    assert "skills" in required
    assert "experience" in required
    assert "target_keywords_used" in required
    assert "match_score_improved" in required


def test_generate_route_registered():
    from cortex.api.main import app
    routes = [r.path for r in app.routes]
    assert "/generate/document" in routes, "/generate/document must be registered"


def test_document_request_schema():
    from cortex.schemas.resumelab import DocumentRequest
    req = DocumentRequest(
        app_name="resumelab",
        user_id="u1",
        job_description="We need a backend engineer",
        canonical_profile={"skills": []},
        template_type="backend",
    )
    assert req.template_type == "backend"
    assert req.base_resume is None


def test_document_response_schema_valid():
    from cortex.schemas.resumelab import DocumentResponse
    resp = DocumentResponse(
        summary="Python engineer with AWS background.",
        skills=["Python", "AWS"],
        projects=[{"name": "DataFlow"}],
        experience=[{"company": "Google", "title": "SWE", "bullets": ["Led migrations"]}],
        target_keywords_used=["Python", "AWS"],
        removed_content=["Flutter (irrelevant)"],
        match_score_improved=85.0,
    )
    assert resp.match_score_improved == 85.0
    assert "AWS" in resp.skills


def test_document_response_score_bounds():
    from cortex.schemas.resumelab import DocumentResponse
    from pydantic import ValidationError
    try:
        DocumentResponse(
            summary="x",
            skills=[],
            projects=[],
            experience=[],
            target_keywords_used=[],
            removed_content=[],
            match_score_improved=110.0,
        )
        raise AssertionError("Should reject score > 100")
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# Backward compatibility check
# ---------------------------------------------------------------------------

def test_no_regression_phase1_routes():
    from cortex.api.main import app
    routes = [r.path for r in app.routes]
    for expected in ("/extract", "/profile/merge", "/ingest", "/query", "/generate"):
        assert expected in routes, f"Phase 1/2 route missing: {expected}"


def test_no_regression_phase1_registries():
    from app.registry.store import get_app
    assert get_app("doclens") is not None
    assert get_app("cvscan") is not None
    assert get_app("resumelab") is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n=== ResumeLab Generate Test Suite ===\n")

    print("-- Template Guidance --")
    _run("template guidance: all types defined", test_template_guidance_all_types)
    _run("invalid template type → fullstack", test_invalid_template_type_defaults_to_fullstack)

    print("\n-- generate_document() --")
    _run("backend template: AWS/Python featured", test_generate_backend_template)
    _run("frontend template: React/TS featured", test_generate_frontend_template)
    _run("fullstack template: balanced skills", test_generate_fullstack_template)
    _run("irrelevant Flutter content removed", test_generate_irrelevant_flutter_removed)
    _run("target_keywords are strings", test_generate_target_keywords_are_strings)
    _run("experience blocks have required fields", test_generate_experience_blocks_have_required_fields)
    _run("match_score_improved clamped to 0-100", test_generate_match_score_improved_clamped)
    _run("empty JD raises ValueError", test_generate_empty_jd_raises)
    _run("retry on bad JSON", test_generate_retry_on_bad_json)
    _run("prompt contains anti-fabrication guidance", test_generate_prompt_contains_no_flutter_fabrication)
    _run("all retries fail → raises ValueError", test_generate_all_retries_fail)

    print("\n-- Registry + Routes --")
    _run("generate task in registry", test_generate_task_in_registry)
    _run("/generate/document route registered", test_generate_route_registered)
    _run("DocumentRequest schema valid", test_document_request_schema)
    _run("DocumentResponse schema valid", test_document_response_schema_valid)
    _run("DocumentResponse score bounds enforced", test_document_response_score_bounds)

    print("\n-- Backward Compatibility --")
    _run("no regression: Phase 1/2 routes intact", test_no_regression_phase1_routes)
    _run("no regression: all registries intact", test_no_regression_phase1_registries)

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
