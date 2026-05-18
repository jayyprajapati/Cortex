"""
ResumeLab profile merge test suite.

Run with:
    python -m scripts.test_resumelab_merge              # all tests (embedding-heavy skipped)
    python -m scripts.test_resumelab_merge --with-embed # run embedding tests too

Tests are grouped:
  Unit      — pure Python, no embeddings (mocked or string-only)
  Embed     — require the embedding model to be loaded (--with-embed flag)
"""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Minimal harness
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
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_embed(texts, model_name="BAAI/bge-small-en"):
    """
    Return deterministic embeddings for predictable tests.

    Items with the same keyword (python, aws, react, ...) share that basis
    dimension, giving them high cosine similarity.  Items with NO keyword hit
    get a unique pseudo-random unit vector seeded from their text hash, so
    unrelated zero-keyword terms don't falsely collide at similarity=1.0.
    """
    vecs = []
    for text in texts:
        base = np.zeros(8, dtype=np.float32)
        for keyword, dim in [
            ("python", 0), ("aws", 1), ("react", 2), ("dataflow", 3),
            ("google", 4), ("stanford", 5), ("aws certified", 6), ("kubernetes", 7),
        ]:
            if keyword in text.lower():
                base[dim] = 1.0
        if base.sum() == 0:
            # No keyword: use a deterministic but unique pseudo-random direction
            seed = abs(hash(text.lower().strip())) % (2 ** 31)
            rng = np.random.RandomState(seed)
            base = rng.randn(8).astype(np.float32)
        norm = np.linalg.norm(base)
        vecs.append(base / norm if norm > 0 else base)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE_A = {
    "doc_id": "doc-001",
    "skills": [
        {"name": "Python", "category": "programming", "proficiency": "expert"},
        {"name": "AWS", "category": "cloud", "proficiency": "advanced"},
        {"name": "React", "category": "frontend", "proficiency": "intermediate"},
        {"name": "Kubernetes", "category": "devops", "proficiency": "proficient"},
    ],
    "projects": [
        {
            "name": "DataFlow",
            "description": "ETL framework",
            "technologies": ["Python", "Airflow"],
            "url": "github.com/jane/dataflow",
            "date_range": "2021",
        }
    ],
    "experience": [
        {
            "company": "Google",
            "title": "Senior Software Engineer",
            "date_range": "2020-Present",
            "location": "Mountain View, CA",
            "bullets": ["Led microservices migration", "Mentored junior engineers"],
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
    "certifications": ["AWS Certified Solutions Architect"],
    "keywords": ["Python", "AWS", "microservices"],
}

# Same person, second resume upload — overlapping data with variations
PROFILE_B = {
    "doc_id": "doc-002",
    "skills": [
        {"name": "Python", "category": "programming", "proficiency": "expert"},  # exact dup
        {"name": "Amazon Web Services", "category": "cloud", "proficiency": "expert"},  # alias
        {"name": "React.js", "category": "frontend", "proficiency": "advanced"},  # alias
        {"name": "Go", "category": "programming", "proficiency": "proficient"},   # new
    ],
    "projects": [
        {
            "name": "DataFlow",  # same project, richer description
            "description": "Open-source distributed ETL framework used by 500+ teams",
            "technologies": ["Python", "Apache Spark", "Airflow"],  # extra tech
            "url": "github.com/jane/dataflow",
            "date_range": "2021-2023",
        }
    ],
    "experience": [
        {
            "company": "Google",
            "title": "Senior Software Engineer",
            "date_range": "2020-Present",  # same dates
            "location": "Mountain View, CA",
            "bullets": [
                "Led microservices migration reducing P95 latency by 40%",  # richer
                "Built internal tooling saving 200 engineer-hours/month",    # new bullet
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
        "AWS Certified Solutions Architect",  # dup
        "Google Cloud Professional Data Engineer",  # new
    ],
    "keywords": ["Python", "Go", "Kubernetes", "ETL"],
}

# Third profile — minimal overlap, mostly new content
PROFILE_C = {
    "doc_id": "doc-003",
    "skills": [
        {"name": "Rust", "category": "programming", "proficiency": "intermediate"},
        {"name": "PostgreSQL", "category": "database", "proficiency": "advanced"},
    ],
    "projects": [
        {
            "name": "RustDB",
            "description": "High-performance database driver in Rust",
            "technologies": ["Rust", "PostgreSQL"],
            "url": None,
            "date_range": "2022",
        }
    ],
    "experience": [
        {
            "company": "Meta",
            "title": "Software Engineer",
            "date_range": "2017-2020",
            "location": "Menlo Park, CA",
            "bullets": ["Built real-time pipeline processing 50M events/day"],
        }
    ],
    "education": [],
    "certifications": [],
    "keywords": ["Rust", "PostgreSQL", "performance"],
}

# Profile with conflicting dates for same experience
PROFILE_DATE_CONFLICT = {
    "doc_id": "doc-004",
    "skills": [],
    "projects": [],
    "experience": [
        {
            "company": "Google",
            "title": "Senior Software Engineer",
            "date_range": "2019-2022",  # DIFFERENT from PROFILE_A's "2020-Present"
            "location": "Mountain View, CA",
            "bullets": ["Optimized distributed query engine"],
        }
    ],
    "education": [],
    "certifications": [],
    "keywords": [],
}

PROFILE_EMPTY = {
    "doc_id": "doc-empty",
    "skills": [],
    "projects": [],
    "experience": [],
    "education": [],
    "certifications": [],
    "keywords": [],
}


# ---------------------------------------------------------------------------
# Unit tests: utilities (no embedding)
# ---------------------------------------------------------------------------

def test_canonical_key():
    from cortex.core.profile_normalizer import _canonical_key
    assert _canonical_key("React.js") == "react_js"
    assert _canonical_key("AWS") == "aws"
    assert _canonical_key("  PostgreSQL  ") == "postgresql"
    assert _canonical_key("Node.js") == "node_js"
    assert _canonical_key("CI/CD") == "ci_cd"


def test_normalize_skill_name():
    from cortex.core.profile_normalizer import _normalize_skill_name
    assert _normalize_skill_name("React.js") == "React"
    assert _normalize_skill_name("amazon web services") == "AWS"
    assert _normalize_skill_name("k8s") == "Kubernetes"
    assert _normalize_skill_name("golang") == "Go"
    assert _normalize_skill_name("PostgreSQL") == "PostgreSQL"  # already canonical
    assert _normalize_skill_name("typescript") == "TypeScript"
    assert _normalize_skill_name("SomeExoticSkill") == "SomeExoticSkill"  # no alias


def test_richer_str():
    from cortex.core.profile_normalizer import _richer_str
    assert _richer_str("short", "a longer string") == "a longer string"
    assert _richer_str("same", "same") == "same"
    assert _richer_str(None, "value") == "value"
    assert _richer_str("value", None) == "value"
    assert _richer_str(None, None) is None


def test_richer_proficiency():
    from cortex.core.profile_normalizer import _richer_proficiency
    assert _richer_proficiency("expert", "intermediate") == "expert"
    assert _richer_proficiency("beginner", "advanced") == "advanced"
    assert _richer_proficiency(None, "proficient") == "proficient"
    assert _richer_proficiency("expert", None) == "expert"
    assert _richer_proficiency(None, None) is None


def test_union_list():
    from cortex.core.profile_normalizer import _union_list
    assert _union_list(["Python", "Go"], ["Go", "Rust"]) == ["Python", "Go", "Rust"]
    assert _union_list([], ["A", "B"]) == ["A", "B"]
    assert _union_list(["X"], []) == ["X"]
    assert _union_list(["a", "b"], ["A", "B"]) == ["a", "b"]  # case-insensitive dedup


def test_greedy_pairs():
    from cortex.core.profile_normalizer import _greedy_pairs
    sim = np.array([[0.95, 0.3], [0.2, 0.88]], dtype=np.float32)
    pairs = _greedy_pairs(sim, threshold=0.8)
    assert len(pairs) == 2
    # Highest pair first: (0,0,0.95) then (1,1,0.88)
    assert pairs[0] == (0, 0, pytest_approx(0.95))
    assert pairs[1] == (1, 1, pytest_approx(0.88))


def pytest_approx(val, rel=1e-3):
    """Float approximate equality helper (no pytest needed)."""
    class Approx:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) < rel
        def __repr__(self): return f"~{self.v}"
    return Approx(val)


def test_greedy_pairs_threshold():
    from cortex.core.profile_normalizer import _greedy_pairs
    sim = np.array([[0.75, 0.6], [0.5, 0.72]], dtype=np.float32)
    pairs = _greedy_pairs(sim, threshold=0.8)
    assert pairs == [], "No pairs above 0.8 threshold"


def test_greedy_pairs_empty():
    from cortex.core.profile_normalizer import _greedy_pairs
    sim = np.zeros((0, 3), dtype=np.float32)
    assert _greedy_pairs(sim, 0.85) == []


def test_source_from_profile():
    from cortex.core.profile_normalizer import _source_from_profile
    assert _source_from_profile({"doc_id": "abc"}) == "abc"
    assert _source_from_profile({"doc_id": "  "}) != ""  # fallback generated
    anon = _source_from_profile({})
    assert anon.startswith("anon_")


# ---------------------------------------------------------------------------
# Unit tests: skill merging (mocked embeddings)
# ---------------------------------------------------------------------------

def test_skill_exact_dedup():
    """Identical skill name should merge into single entry."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_skill_lists
        ex = [{"name": "Python", "category": "programming", "proficiency": "expert"}]
        inc = [{"name": "Python", "category": "programming", "proficiency": "intermediate"}]
        result, added, dups = _merge_skill_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1, f"Should merge to 1 skill, got {len(result)}"
    assert result[0]["normalized_name"] == "Python"
    assert result[0]["proficiency"] == "expert"  # richer wins
    assert "doc-a" in result[0]["sources"]
    assert "doc-b" in result[0]["sources"]
    assert len(dups) == 1, "Should record one duplicate merge"
    assert added == []


def test_skill_alias_dedup():
    """'amazon web services' should normalize to 'AWS' and merge with 'AWS'."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_skill_lists
        ex = [{"name": "AWS", "category": "cloud", "proficiency": "advanced"}]
        inc = [{"name": "Amazon Web Services", "category": "cloud", "proficiency": "expert"}]
        result, added, dups = _merge_skill_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1, f"AWS aliases must merge. Got: {[r['normalized_name'] for r in result]}"
    assert result[0]["proficiency"] == "expert"
    assert len(result[0]["aliases"]) >= 2, "Both name forms should be recorded as aliases"
    assert added == []


def test_skill_alias_react():
    """'React.js' should normalize to 'React' and merge."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_skill_lists
        ex = [{"name": "React", "category": "frontend", "proficiency": "intermediate"}]
        inc = [{"name": "React.js", "category": "frontend", "proficiency": "advanced"}]
        result, added, dups = _merge_skill_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1
    assert result[0]["proficiency"] == "advanced"
    assert "React" in result[0]["aliases"]
    assert "React.js" in result[0]["aliases"]


def test_skill_no_overlap():
    """Completely different skills should all be preserved."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_skill_lists
        ex = [{"name": "Python", "category": None, "proficiency": None}]
        inc = [{"name": "Rust", "category": None, "proficiency": None}]
        result, added, dups = _merge_skill_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 2
    names = {r["normalized_name"] for r in result}
    assert "Python" in names
    assert "Rust" in names
    assert len(added) == 1
    assert dups == []


def test_skill_sources_preserved():
    """Source doc_id must be tracked on each canonical skill."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_skill_lists
        ex = [{"name": "Go", "category": None, "proficiency": None}]
        inc = [{"name": "Go", "category": None, "proficiency": None}]
        result, _, _ = _merge_skill_lists(ex, inc, "doc-X", "doc-Y", 0.85, "mock")

    assert "doc-X" in result[0]["sources"]
    assert "doc-Y" in result[0]["sources"]


# ---------------------------------------------------------------------------
# Unit tests: project merging (mocked embeddings)
# ---------------------------------------------------------------------------

def test_project_exact_dedup():
    """Same project name should merge, combining technologies."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_project_lists
        ex = [{"name": "DataFlow", "description": "ETL framework", "technologies": ["Python"], "url": None, "date_range": None}]
        inc = [{"name": "DataFlow", "description": "Open-source distributed ETL framework", "technologies": ["Python", "Spark"], "url": "github.com/x/df", "date_range": "2021"}]
        result, added, dups = _merge_project_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1
    assert "Spark" in result[0]["technologies"]
    assert "Python" in result[0]["technologies"]
    assert result[0]["url"] == "github.com/x/df"
    assert "Open-source" in result[0]["description"]  # richer description wins


def test_project_distinct_not_merged():
    """Two clearly different projects should NOT be merged."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_project_lists
        ex = [{"name": "DataFlow", "description": "ETL pipeline tool", "technologies": ["Python"], "url": None, "date_range": None}]
        inc = [{"name": "RustDB", "description": "Database driver in Rust", "technologies": ["Rust"], "url": None, "date_range": None}]
        result, added, dups = _merge_project_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 2
    names = {r["normalized_name"] for r in result}
    assert "DataFlow" in names
    assert "RustDB" in names
    assert len(added) == 1
    assert dups == []


# ---------------------------------------------------------------------------
# Unit tests: experience merging
# ---------------------------------------------------------------------------

def test_experience_same_company_title_merges():
    """Same company+title entry should merge with bullet dedup."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_experience_lists
        ex = [{"company": "Google", "title": "SWE", "date_range": "2020-Present",
               "location": "MTV", "bullets": ["Led migrations"]}]
        inc = [{"company": "Google", "title": "SWE", "date_range": "2020-Present",
                "location": "MTV", "bullets": ["Led migrations reducing latency 40%", "New bullet"]}]
        result, added, dups, conflicts = _merge_experience_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1
    assert len(result[0]["bullets"]) >= 2  # at least "Led migrations" + "New bullet"
    assert conflicts == []
    assert len(dups) == 1


def test_experience_date_conflict_flagged():
    """Matching experience with different dates should raise a conflict."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_experience_lists
        ex = [{"company": "Google", "title": "SWE", "date_range": "2020-Present",
               "location": None, "bullets": []}]
        inc = [{"company": "Google", "title": "SWE", "date_range": "2019-2022",
                "location": None, "bullets": []}]
        result, _, _, conflicts = _merge_experience_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["type"] == "date_range_conflict"
    assert conflict["existing_date"] == "2020-Present"
    assert conflict["incoming_date"] == "2019-2022"
    assert conflict["resolution"] == "kept_existing"
    assert result[0]["date_range"] == "2020-Present"  # existing kept


def test_experience_no_overlap_both_preserved():
    """Two completely different jobs should both appear in the result."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_experience_lists
        ex = [{"company": "Google", "title": "SWE", "date_range": "2020-Present",
               "location": None, "bullets": []}]
        inc = [{"company": "Meta", "title": "Engineer", "date_range": "2017-2020",
                "location": None, "bullets": []}]
        result, added, _, _ = _merge_experience_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 2
    companies = {r["company"] for r in result}
    assert "Google" in companies
    assert "Meta" in companies
    assert len(added) == 1


# ---------------------------------------------------------------------------
# Unit tests: education merging
# ---------------------------------------------------------------------------

def test_education_dedup():
    """Same institution+degree should merge."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_education_lists
        ex = [{"institution": "Stanford University", "degree": "B.S.", "field_of_study": "CS", "date_range": "2017", "gpa": "3.9"}]
        inc = [{"institution": "Stanford University", "degree": "B.S.", "field_of_study": "Computer Science", "date_range": "2017", "gpa": None}]
        result, added, dups = _merge_education_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 1
    assert result[0]["gpa"] == "3.9"  # non-null wins
    assert result[0]["field_of_study"] == "Computer Science"  # richer wins
    assert len(dups) == 1


# ---------------------------------------------------------------------------
# Unit tests: certification merging
# ---------------------------------------------------------------------------

def test_cert_exact_dedup():
    """Exact same cert should merge."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import _merge_cert_lists
        ex = ["AWS Certified Solutions Architect"]
        inc = ["AWS Certified Solutions Architect", "Google Cloud Professional Data Engineer"]
        result, added, dups = _merge_cert_lists(ex, inc, "doc-a", "doc-b", 0.85, "mock")

    assert len(result) == 2  # 1 merged + 1 new
    names = {r["normalized_name"] for r in result}
    assert "AWS Certified Solutions Architect" in names
    assert "Google Cloud Professional Data Engineer" in names
    assert len(added) == 1
    assert len(dups) == 1


# ---------------------------------------------------------------------------
# Integration tests: merge_profiles (full pipeline, mocked embeddings)
# ---------------------------------------------------------------------------

def test_merge_profiles_ab_basic():
    """Merge Profile A and B: 3 skills deduplicated, 1 project merged, 1 experience merged."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_B, threshold=0.85)

    cp = result["canonical_profile"]
    assert "skills" in cp
    assert "projects" in cp
    assert "experience" in cp
    assert "source_documents" in cp

    skill_names = {s["normalized_name"] for s in cp["skills"]}
    assert "Python" in skill_names
    assert "AWS" in skill_names
    assert "React" in skill_names or "React.js" in skill_names
    assert "Go" in skill_names  # new from B

    # Python should appear once despite being in both
    python_entries = [s for s in cp["skills"] if s["normalized_name"] == "Python"]
    assert len(python_entries) == 1, "Python must be deduplicated"

    # AWS should appear once (Amazon Web Services alias)
    aws_entries = [s for s in cp["skills"] if s["normalized_name"] == "AWS"]
    assert len(aws_entries) == 1, "AWS / Amazon Web Services must deduplicate"

    # DataFlow should appear once
    df_entries = [p for p in cp["projects"] if "DataFlow" in p["normalized_name"]]
    assert len(df_entries) == 1, "DataFlow must deduplicate"
    assert "Apache Spark" in df_entries[0]["technologies"]  # merged from B

    # Source docs tracked
    assert "doc-001" in cp["source_documents"]
    assert "doc-002" in cp["source_documents"]


def test_merge_profiles_stats():
    """Stats must correctly report before/after counts."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_B, threshold=0.85)

    stats = result["stats"]
    assert "skills_before" in stats
    assert "skills_after" in stats
    assert stats["skills_before"] == len(PROFILE_A["skills"]) + len(PROFILE_B["skills"])
    assert stats["skills_after"] <= stats["skills_before"], "Merging can only reduce or equal count"
    assert stats["projects_before"] == 2
    assert stats["projects_after"] == 1  # DataFlow deduplicated


def test_merge_profiles_three_way():
    """Merge A+B then merge with C: C's unique items added, no data lost."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        ab = merge_profiles(PROFILE_A, PROFILE_B, threshold=0.85)
        abc = merge_profiles(ab["canonical_profile"], PROFILE_C, threshold=0.85)

    cp = abc["canonical_profile"]
    skill_names = {s["normalized_name"] for s in cp["skills"]}
    assert "Rust" in skill_names  # new from C
    assert "PostgreSQL" in skill_names  # new from C

    project_names = {p["normalized_name"] for p in cp["projects"]}
    assert "RustDB" in project_names  # new from C

    companies = {e["company"] for e in cp["experience"]}
    assert "Google" in companies
    assert "Meta" in companies  # from C


def test_merge_profiles_date_conflict_flagged():
    """Date conflict in experience should appear in conflicts, merge still succeeds."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_DATE_CONFLICT, threshold=0.85)

    conflicts = result["conflicts"].get("experience", [])
    assert len(conflicts) == 1
    assert conflicts[0]["type"] == "date_range_conflict"
    assert conflicts[0]["resolution"] == "kept_existing"

    # Canonical profile still produced
    cp = result["canonical_profile"]
    assert cp["experience"][0]["date_range"] == "2020-Present"


def test_merge_profiles_zero_overlap():
    """A and C have no overlapping skills/projects/experience. All items preserved."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_C, threshold=0.85)

    cp = result["canonical_profile"]
    assert len(cp["skills"]) == len(PROFILE_A["skills"]) + len(PROFILE_C["skills"])
    assert len(cp["projects"]) == len(PROFILE_A["projects"]) + len(PROFILE_C["projects"])

    added = result["added_items"]
    assert len(added["skills"]) == len(PROFILE_C["skills"])
    assert result["merged_duplicates"]["skills"] == []


def test_merge_profiles_existing_empty():
    """Merging empty existing + non-empty incoming returns incoming as canonical."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_EMPTY, PROFILE_A, threshold=0.85)

    cp = result["canonical_profile"]
    assert len(cp["skills"]) == len(PROFILE_A["skills"])
    assert len(cp["experience"]) == len(PROFILE_A["experience"])
    assert len(cp["certifications"]) == len(PROFILE_A["certifications"])


def test_merge_profiles_incoming_empty():
    """Merging non-empty existing + empty incoming returns existing unchanged."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_EMPTY, threshold=0.85)

    cp = result["canonical_profile"]
    assert len(cp["skills"]) == len(PROFILE_A["skills"])


def test_merge_preserves_all_keywords():
    """Keywords from both profiles must all appear in merged output."""
    with patch("cortex.core.profile_normalizer._embed_texts", side_effect=_mock_embed):
        from cortex.core.profile_normalizer import merge_profiles
        result = merge_profiles(PROFILE_A, PROFILE_B, threshold=0.85)

    merged_kws = set(result["canonical_profile"]["keywords"])
    for kw in PROFILE_A["keywords"] + PROFILE_B["keywords"]:
        assert kw in merged_kws, f"Keyword {kw!r} lost during merge"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_canonical_profile_schema():
    """CanonicalProfile Pydantic model validates correctly."""
    from cortex.schemas.resumelab import CanonicalProfile, CanonicalSkill
    from datetime import datetime, timezone
    cp = CanonicalProfile(
        skills=[CanonicalSkill(canonical_key="python", normalized_name="Python",
                               sources=["doc-001"])],
        merged_at=datetime.now(timezone.utc).isoformat(),
    )
    assert cp.skills[0].canonical_key == "python"
    assert cp.keywords == []


def test_profile_merge_request_schema():
    """ProfileMergeRequest validates threshold bounds."""
    from cortex.schemas.resumelab import ProfileMergeRequest
    from pydantic import ValidationError
    req = ProfileMergeRequest(
        app_name="resumelab",
        user_id="u1",
        existing_profile={},
        incoming_profile={},
        similarity_threshold=0.85,
    )
    assert req.similarity_threshold == 0.85

    try:
        ProfileMergeRequest(
            app_name="resumelab", user_id="u1",
            existing_profile={}, incoming_profile={},
            similarity_threshold=1.5,  # out of range
        )
        raise AssertionError("Should reject threshold > 1.0")
    except ValidationError:
        pass


def test_canonical_skill_confidence_bounds():
    from cortex.schemas.resumelab import CanonicalSkill
    from pydantic import ValidationError
    try:
        CanonicalSkill(canonical_key="x", normalized_name="X", confidence=1.5)
        raise AssertionError("Should reject confidence > 1.0")
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# FastAPI route test
# ---------------------------------------------------------------------------

def test_profile_merge_route_registered():
    from cortex.api.main import app
    routes = [r.path for r in app.routes]
    assert "/profile/merge" in routes, "/profile/merge must be registered"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(with_embed: bool = False) -> int:
    print("\n=== ResumeLab Profile Merge Test Suite ===\n")

    print("-- Utilities --")
    _run("canonical_key", test_canonical_key)
    _run("normalize_skill_name", test_normalize_skill_name)
    _run("richer_str", test_richer_str)
    _run("richer_proficiency", test_richer_proficiency)
    _run("union_list", test_union_list)
    _run("greedy_pairs", test_greedy_pairs)
    _run("greedy_pairs threshold", test_greedy_pairs_threshold)
    _run("greedy_pairs empty", test_greedy_pairs_empty)
    _run("source_from_profile", test_source_from_profile)

    print("\n-- Skill Merging --")
    _run("skill: exact dedup", test_skill_exact_dedup)
    _run("skill: alias AWS dedup", test_skill_alias_dedup)
    _run("skill: alias React dedup", test_skill_alias_react)
    _run("skill: no overlap preserved", test_skill_no_overlap)
    _run("skill: sources tracked", test_skill_sources_preserved)

    print("\n-- Project Merging --")
    _run("project: exact dedup + tech union", test_project_exact_dedup)
    _run("project: distinct not merged", test_project_distinct_not_merged)

    print("\n-- Experience Merging --")
    _run("experience: same job merges", test_experience_same_company_title_merges)
    _run("experience: date conflict flagged", test_experience_date_conflict_flagged)
    _run("experience: no overlap preserved", test_experience_no_overlap_both_preserved)

    print("\n-- Education Merging --")
    _run("education: dedup", test_education_dedup)

    print("\n-- Certification Merging --")
    _run("cert: exact dedup", test_cert_exact_dedup)

    print("\n-- merge_profiles() Integration --")
    _run("A+B basic merge", test_merge_profiles_ab_basic)
    _run("A+B stats", test_merge_profiles_stats)
    _run("three-way merge A+B+C", test_merge_profiles_three_way)
    _run("date conflict flagged", test_merge_profiles_date_conflict_flagged)
    _run("zero overlap: all items preserved", test_merge_profiles_zero_overlap)
    _run("empty existing: incoming becomes canonical", test_merge_profiles_existing_empty)
    _run("empty incoming: existing unchanged", test_merge_profiles_incoming_empty)
    _run("keywords union preserved", test_merge_preserves_all_keywords)

    print("\n-- Schemas --")
    _run("CanonicalProfile schema valid", test_canonical_profile_schema)
    _run("ProfileMergeRequest threshold bounds", test_profile_merge_request_schema)
    _run("CanonicalSkill confidence bounds", test_canonical_skill_confidence_bounds)

    print("\n-- Route Registration --")
    _run("/profile/merge route registered", test_profile_merge_route_registered)

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
    parser = argparse.ArgumentParser(description="ResumeLab merge test suite")
    parser.add_argument("--with-embed", action="store_true",
                        help="Run tests that require real embedding model")
    args = parser.parse_args()
    sys.exit(main(with_embed=args.with_embed))
