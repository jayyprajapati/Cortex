"""
profile_normalizer.py

Canonical profile merging and deduplication engine.

Merges two profile dicts (ExtractResponse shape or CanonicalProfile shape) into
a single CanonicalProfile with full source traceability.

Algorithm per section:
  Phase 1  — normalize names via alias map, build canonical_key
  Phase 2  — exact match on canonical_key
  Phase 3  — embedding similarity for remaining unmatched items (greedy bipartite)
  Merge    — richer-wins heuristic per field; union for list fields
  Conflict — flag when matched items have contradictory required fields (e.g. dates)

No Qdrant writes.  The caller owns storage decisions.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill name normalization map
# ---------------------------------------------------------------------------

_SKILL_NORM: Dict[str, str] = {
    # JavaScript ecosystem
    "react.js": "React",
    "reactjs": "React",
    "react js": "React",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "express.js": "Express.js",
    "expressjs": "Express.js",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "angular.js": "Angular",
    "angularjs": "Angular",
    "angular js": "Angular",
    "nuxt.js": "Nuxt.js",
    "nuxtjs": "Nuxt.js",
    # Cloud providers
    "amazon web services": "AWS",
    "amazon aws": "AWS",
    "aws cloud": "AWS",
    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "gcp": "GCP",
    "microsoft azure": "Azure",
    "azure cloud": "Azure",
    # Databases
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "mongo db": "MongoDB",
    "mysql": "MySQL",
    "my sql": "MySQL",
    "mssql": "SQL Server",
    "microsoft sql server": "SQL Server",
    "elastic search": "Elasticsearch",
    # Languages
    "python3": "Python",
    "python 3": "Python",
    "golang": "Go",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "ts": "TypeScript",
    "c++": "C++",
    "cplusplus": "C++",
    "c sharp": "C#",
    # DevOps / infra
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "ci cd": "CI/CD",
    "continuous integration": "CI/CD",
    "terraform": "Terraform",
    "ansible": "Ansible",
    # ML / AI
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "natural language processing": "NLP",
    "large language models": "LLMs",
    "llm": "LLMs",
    "deep learning": "Deep Learning",
    "neural networks": "Neural Networks",
    # APIs
    "restful api": "REST API",
    "restful apis": "REST API",
    "rest": "REST API",
    "rest apis": "REST API",
    "graphql": "GraphQL",
    # Misc
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "linux": "Linux",
    "unix": "Unix",
}

_PROFICIENCY_RANK: Dict[str, int] = {
    "expert": 5,
    "advanced": 4,
    "proficient": 3,
    "intermediate": 2,
    "beginner": 1,
    "basic": 1,
    "familiar": 1,
}

# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _canonical_key(text: str) -> str:
    """Stable lowercase slug for dedup comparisons."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _normalize_skill_name(name: str) -> str:
    key = name.strip().lower()
    return _SKILL_NORM.get(key, name.strip())


def _richer_str(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Return whichever string has more content; neither beats None."""
    sa, sb = (a or ""), (b or "")
    return a if len(sa) >= len(sb) else (b or a)


def _richer_proficiency(a: Optional[str], b: Optional[str]) -> Optional[str]:
    ra = _PROFICIENCY_RANK.get((a or "").strip().lower(), 0)
    rb = _PROFICIENCY_RANK.get((b or "").strip().lower(), 0)
    if ra == 0 and rb == 0:
        return a or b
    return a if ra >= rb else b


def _union_list(a: List[str], b: List[str]) -> List[str]:
    """Deduplicated union preserving order (a first, then new items from b)."""
    seen = {x.strip().lower() for x in a}
    result = list(a)
    for item in b:
        if item.strip().lower() not in seen:
            result.append(item)
            seen.add(item.strip().lower())
    return result


def _source_from_profile(profile: Dict[str, Any]) -> str:
    """Extract the doc_id from a profile dict or generate a fallback."""
    doc_id = str(profile.get("doc_id") or "").strip()
    return doc_id or f"anon_{uuid.uuid4().hex[:8]}"


def _sources_from_profile(profile: Dict[str, Any]) -> List[str]:
    """Return list of source doc_ids. Handles both ExtractResponse and CanonicalProfile."""
    explicit = profile.get("source_documents")
    if explicit and isinstance(explicit, list):
        return [s for s in explicit if s]
    single = _source_from_profile(profile)
    return [single] if single else []


def _merge_sources(a: List[str], b: List[str]) -> List[str]:
    return _union_list(a, b)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """
    Embed a list of texts using the project's shared model cache.
    Returns an (N, D) array of L2-normalised vectors.
    """
    from app.embeddings.embedder import _get_model  # reuse model cache
    model = _get_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


def _sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity matrix (A rows vs B rows).
    Both A, B must be L2-normalised → dot product = cosine similarity.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return (A @ B.T).astype(np.float32)


def _greedy_pairs(
    sim: np.ndarray, threshold: float
) -> List[Tuple[int, int, float]]:
    """
    Greedy bipartite matching: take highest-sim pairs above threshold,
    ensuring each index appears at most once on each side.
    """
    if sim.size == 0:
        return []
    rows, cols = np.where(sim >= threshold)
    if rows.size == 0:
        return []
    scores = sorted(
        zip(sim[rows, cols].tolist(), rows.tolist(), cols.tolist()),
        reverse=True,
    )
    matched_a: set = set()
    matched_b: set = set()
    pairs: List[Tuple[int, int, float]] = []
    for score, i, j in scores:
        if i not in matched_a and j not in matched_b:
            pairs.append((int(i), int(j), float(score)))
            matched_a.add(i)
            matched_b.add(j)
    return pairs


# ---------------------------------------------------------------------------
# Input coercion helpers (handle both raw and canonical shapes)
# ---------------------------------------------------------------------------

def _coerce_skill(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        name = str(item.get("normalized_name") or item.get("name") or "").strip()
        return {
            "name": name,
            "category": item.get("category"),
            "proficiency": item.get("proficiency"),
            "_existing_aliases": item.get("aliases", []),
            "_existing_sources": item.get("sources", []),
            "_existing_key": item.get("canonical_key"),
        } if name else None
    if isinstance(item, str) and item.strip():
        return {"name": item.strip(), "category": None, "proficiency": None,
                "_existing_aliases": [], "_existing_sources": [], "_existing_key": None}
    return None


def _coerce_project(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    name = str(item.get("normalized_name") or item.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(item.get("description") or ""),
        "technologies": list(item.get("technologies") or []),
        "url": item.get("url"),
        "date_range": item.get("date_range"),
        "_existing_aliases": item.get("aliases", []),
        "_existing_sources": item.get("sources", []),
        "_existing_key": item.get("canonical_key"),
    }


def _coerce_experience(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    company = str(item.get("company") or "").strip()
    title = str(item.get("title") or "").strip()
    if not company and not title:
        return None
    return {
        "company": company,
        "title": title,
        "date_range": item.get("date_range"),
        "location": item.get("location"),
        "bullets": list(item.get("bullets") or []),
        "_existing_sources": item.get("sources", []),
        "_existing_key": item.get("canonical_key"),
    }


def _coerce_education(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    institution = str(item.get("institution") or "").strip()
    if not institution:
        return None
    return {
        "institution": institution,
        "degree": item.get("degree"),
        "field_of_study": item.get("field_of_study"),
        "date_range": item.get("date_range"),
        "gpa": item.get("gpa"),
        "_existing_sources": item.get("sources", []),
        "_existing_key": item.get("canonical_key"),
    }


def _coerce_cert(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        name = str(item.get("normalized_name") or item.get("name") or "").strip()
        return name or None
    return None


# ---------------------------------------------------------------------------
# Per-section merge functions
# ---------------------------------------------------------------------------

def _merge_skill_lists(
    existing: List[Any],
    incoming: List[Any],
    src_ex: str,
    src_inc: str,
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns (canonical_list, added, merged_records).
    merged_records: [{existing, incoming, result, similarity?}]
    """
    def _build(item: Dict, source: str) -> Dict:
        raw_name = item["name"]
        norm_name = _normalize_skill_name(raw_name)
        existing_aliases = item.get("_existing_aliases") or []
        existing_sources = item.get("_existing_sources") or []
        aliases = list(dict.fromkeys([raw_name, norm_name] + existing_aliases))
        return {
            "canonical_key": item.get("_existing_key") or _canonical_key(norm_name),
            "normalized_name": norm_name,
            "aliases": aliases,
            "category": item.get("category"),
            "proficiency": item.get("proficiency"),
            "sources": _merge_sources(existing_sources, [source] if source not in existing_sources else []),
            "confidence": 1.0,
        }

    ex_items = [_build(s, src_ex) for s in ((_coerce_skill(x) for x in existing) if True else []) if s]
    ex_items = [x for x in ex_items if x]
    inc_items = [_build(s, src_inc) for s in ((_coerce_skill(x) for x in incoming) if True else []) if s]
    inc_items = [x for x in inc_items if x]

    ex_by_key = {s["canonical_key"]: i for i, s in enumerate(ex_items)}
    matched_ex: Dict[int, int] = {}
    matched_inc: Dict[int, int] = {}

    for j, inc in enumerate(inc_items):
        if inc["canonical_key"] in ex_by_key:
            i = ex_by_key[inc["canonical_key"]]
            matched_ex[i] = j
            matched_inc[j] = i

    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    unmatched_inc = [j for j in range(len(inc_items)) if j not in matched_inc]

    if unmatched_ex and unmatched_inc:
        ex_texts = [ex_items[i]["normalized_name"] for i in unmatched_ex]
        inc_texts = [inc_items[j]["normalized_name"] for j in unmatched_inc]
        sim = _sim_matrix(
            _embed_texts(ex_texts, model_name),
            _embed_texts(inc_texts, model_name),
        )
        for i_local, j_local, _ in _greedy_pairs(sim, threshold):
            i_global = unmatched_ex[i_local]
            j_global = unmatched_inc[j_local]
            matched_ex[i_global] = j_global
            matched_inc[j_global] = i_global

    merged_records: List[Dict] = []
    result: List[Dict] = []
    processed_inc: set = set()

    for i, ex in enumerate(ex_items):
        if i in matched_ex:
            j = matched_ex[i]
            inc = inc_items[j]
            merged = _merge_two_skills(ex, inc)
            merged_records.append({"existing": ex, "incoming": inc, "result": merged})
            result.append(merged)
            processed_inc.add(j)
        else:
            result.append(ex)

    added = []
    for j, inc in enumerate(inc_items):
        if j not in processed_inc:
            result.append(inc)
            added.append(inc)

    return result, added, merged_records


def _merge_two_skills(a: Dict, b: Dict) -> Dict:
    return {
        "canonical_key": a["canonical_key"],
        "normalized_name": a["normalized_name"],
        "aliases": list(dict.fromkeys(a.get("aliases", []) + b.get("aliases", []))),
        "category": a.get("category") or b.get("category"),
        "proficiency": _richer_proficiency(a.get("proficiency"), b.get("proficiency")),
        "sources": _merge_sources(a.get("sources", []), b.get("sources", [])),
        "confidence": max(a.get("confidence", 1.0), b.get("confidence", 1.0)),
    }


def _merge_project_lists(
    existing: List[Any],
    incoming: List[Any],
    src_ex: str,
    src_inc: str,
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    def _build(item: Dict, source: str) -> Dict:
        name = item["name"]
        existing_aliases = item.get("_existing_aliases") or []
        existing_sources = item.get("_existing_sources") or []
        aliases = list(dict.fromkeys([name] + existing_aliases))
        return {
            "canonical_key": item.get("_existing_key") or _canonical_key(name),
            "normalized_name": name,
            "aliases": aliases,
            "description": item.get("description") or "",
            "technologies": list(item.get("technologies") or []),
            "url": item.get("url"),
            "date_range": item.get("date_range"),
            "sources": _merge_sources(existing_sources, [source] if source not in existing_sources else []),
            "confidence": 1.0,
        }

    ex_items = [_build(p, src_ex) for p in (_coerce_project(x) for x in existing) if p]
    inc_items = [_build(p, src_inc) for p in (_coerce_project(x) for x in incoming) if p]

    ex_by_key = {p["canonical_key"]: i for i, p in enumerate(ex_items)}
    matched_ex: Dict[int, int] = {}
    matched_inc: Dict[int, int] = {}

    for j, inc in enumerate(inc_items):
        if inc["canonical_key"] in ex_by_key:
            i = ex_by_key[inc["canonical_key"]]
            matched_ex[i] = j
            matched_inc[j] = i

    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    unmatched_inc = [j for j in range(len(inc_items)) if j not in matched_inc]

    if unmatched_ex and unmatched_inc:
        # Embed name + description for richer matching signal
        ex_texts = [
            f"{ex_items[i]['normalized_name']} {ex_items[i]['description']}"
            for i in unmatched_ex
        ]
        inc_texts = [
            f"{inc_items[j]['normalized_name']} {inc_items[j]['description']}"
            for j in unmatched_inc
        ]
        sim = _sim_matrix(
            _embed_texts(ex_texts, model_name),
            _embed_texts(inc_texts, model_name),
        )
        for i_local, j_local, score in _greedy_pairs(sim, threshold):
            i_global = unmatched_ex[i_local]
            j_global = unmatched_inc[j_local]
            matched_ex[i_global] = j_global
            matched_inc[j_global] = i_global

    merged_records: List[Dict] = []
    result: List[Dict] = []
    processed_inc: set = set()

    for i, ex in enumerate(ex_items):
        if i in matched_ex:
            j = matched_ex[i]
            inc = inc_items[j]
            merged = _merge_two_projects(ex, inc)
            merged_records.append({"existing": ex, "incoming": inc, "result": merged})
            result.append(merged)
            processed_inc.add(j)
        else:
            result.append(ex)

    added = []
    for j, inc in enumerate(inc_items):
        if j not in processed_inc:
            result.append(inc)
            added.append(inc)

    return result, added, merged_records


def _merge_two_projects(a: Dict, b: Dict) -> Dict:
    return {
        "canonical_key": a["canonical_key"],
        "normalized_name": a["normalized_name"],
        "aliases": list(dict.fromkeys(a.get("aliases", []) + b.get("aliases", []))),
        "description": _richer_str(a.get("description"), b.get("description")) or "",
        "technologies": _union_list(
            list(a.get("technologies") or []),
            list(b.get("technologies") or []),
        ),
        "url": a.get("url") or b.get("url"),
        "date_range": _richer_str(a.get("date_range"), b.get("date_range")),
        "sources": _merge_sources(a.get("sources", []), b.get("sources", [])),
        "confidence": max(a.get("confidence", 1.0), b.get("confidence", 1.0)),
    }


def _merge_experience_lists(
    existing: List[Any],
    incoming: List[Any],
    src_ex: str,
    src_inc: str,
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Returns (canonical_list, added, merged_records, conflicts)."""

    def _build(item: Dict, source: str) -> Dict:
        company = item["company"]
        title = item["title"]
        key = item.get("_existing_key") or _canonical_key(f"{company}_{title}")
        existing_sources = item.get("_existing_sources") or []
        return {
            "canonical_key": key,
            "company": company,
            "title": title,
            "date_range": item.get("date_range"),
            "location": item.get("location"),
            "bullets": list(item.get("bullets") or []),
            "sources": _merge_sources(existing_sources, [source] if source not in existing_sources else []),
            "confidence": 1.0,
        }

    ex_items = [_build(e, src_ex) for e in (_coerce_experience(x) for x in existing) if e]
    inc_items = [_build(e, src_inc) for e in (_coerce_experience(x) for x in incoming) if e]

    ex_by_key = {e["canonical_key"]: i for i, e in enumerate(ex_items)}
    matched_ex: Dict[int, int] = {}
    matched_inc: Dict[int, int] = {}

    for j, inc in enumerate(inc_items):
        if inc["canonical_key"] in ex_by_key:
            i = ex_by_key[inc["canonical_key"]]
            matched_ex[i] = j
            matched_inc[j] = i

    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    unmatched_inc = [j for j in range(len(inc_items)) if j not in matched_inc]

    if unmatched_ex and unmatched_inc:
        ex_texts = [
            f"{ex_items[i]['company']} {ex_items[i]['title']}"
            for i in unmatched_ex
        ]
        inc_texts = [
            f"{inc_items[j]['company']} {inc_items[j]['title']}"
            for j in unmatched_inc
        ]
        sim = _sim_matrix(
            _embed_texts(ex_texts, model_name),
            _embed_texts(inc_texts, model_name),
        )
        for i_local, j_local, _ in _greedy_pairs(sim, threshold):
            i_global = unmatched_ex[i_local]
            j_global = unmatched_inc[j_local]
            matched_ex[i_global] = j_global
            matched_inc[j_global] = i_global

    merged_records: List[Dict] = []
    conflicts: List[Dict] = []
    result: List[Dict] = []
    processed_inc: set = set()

    for i, ex in enumerate(ex_items):
        if i in matched_ex:
            j = matched_ex[i]
            inc = inc_items[j]
            merged, conflict = _merge_two_experiences(ex, inc, model_name)
            merged_records.append({"existing": ex, "incoming": inc, "result": merged})
            if conflict:
                conflicts.append(conflict)
            result.append(merged)
            processed_inc.add(j)
        else:
            result.append(ex)

    added = []
    for j, inc in enumerate(inc_items):
        if j not in processed_inc:
            result.append(inc)
            added.append(inc)

    return result, added, merged_records, conflicts


def _merge_bullets(ex_bullets: List[str], inc_bullets: List[str], model_name: str) -> List[str]:
    """
    Union of bullets with semantic dedup: if an incoming bullet is very similar
    to an existing one, keep the richer (longer) version rather than adding a duplicate.
    """
    if not inc_bullets:
        return list(ex_bullets)
    if not ex_bullets:
        return list(inc_bullets)

    sim = _sim_matrix(
        _embed_texts(ex_bullets, model_name),
        _embed_texts(inc_bullets, model_name),
    )
    # Higher threshold for bullet dedup — bullets need to be very similar to count as duplicate
    _BULLET_THRESHOLD = 0.92
    matched_inc: set = set()
    result = list(ex_bullets)

    pairs = _greedy_pairs(sim, _BULLET_THRESHOLD)
    for i_ex, j_inc, _ in pairs:
        existing_bullet = result[i_ex]
        incoming_bullet = inc_bullets[j_inc]
        if len(incoming_bullet) > len(existing_bullet):
            result[i_ex] = incoming_bullet  # richer bullet wins
        matched_inc.add(j_inc)

    for j, bullet in enumerate(inc_bullets):
        if j not in matched_inc:
            result.append(bullet)

    return result


def _merge_two_experiences(
    a: Dict, b: Dict, model_name: str
) -> Tuple[Dict, Optional[Dict]]:
    """Returns (merged, conflict_or_None)."""
    conflict = None
    date_range = a.get("date_range") or b.get("date_range")

    if (
        a.get("date_range")
        and b.get("date_range")
        and a["date_range"] != b["date_range"]
    ):
        conflict = {
            "type": "date_range_conflict",
            "canonical_key": a["canonical_key"],
            "company": a["company"],
            "title": a["title"],
            "existing_date": a["date_range"],
            "incoming_date": b["date_range"],
            "resolution": "kept_existing",
        }
        date_range = a["date_range"]  # existing is authoritative

    merged_bullets = _merge_bullets(
        list(a.get("bullets") or []),
        list(b.get("bullets") or []),
        model_name,
    )

    return {
        "canonical_key": a["canonical_key"],
        "company": a["company"],
        "title": a["title"],
        "date_range": date_range,
        "location": _richer_str(a.get("location"), b.get("location")),
        "bullets": merged_bullets,
        "sources": _merge_sources(a.get("sources", []), b.get("sources", [])),
        "confidence": max(a.get("confidence", 1.0), b.get("confidence", 1.0)),
    }, conflict


def _merge_education_lists(
    existing: List[Any],
    incoming: List[Any],
    src_ex: str,
    src_inc: str,
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    def _build(item: Dict, source: str) -> Dict:
        institution = item["institution"]
        degree = item.get("degree") or ""
        key = item.get("_existing_key") or _canonical_key(f"{institution}_{degree}")
        existing_sources = item.get("_existing_sources") or []
        return {
            "canonical_key": key,
            "institution": institution,
            "degree": item.get("degree"),
            "field_of_study": item.get("field_of_study"),
            "date_range": item.get("date_range"),
            "gpa": item.get("gpa"),
            "sources": _merge_sources(existing_sources, [source] if source not in existing_sources else []),
            "confidence": 1.0,
        }

    ex_items = [_build(e, src_ex) for e in (_coerce_education(x) for x in existing) if e]
    inc_items = [_build(e, src_inc) for e in (_coerce_education(x) for x in incoming) if e]

    ex_by_key = {e["canonical_key"]: i for i, e in enumerate(ex_items)}
    matched_ex: Dict[int, int] = {}
    matched_inc: Dict[int, int] = {}

    for j, inc in enumerate(inc_items):
        if inc["canonical_key"] in ex_by_key:
            i = ex_by_key[inc["canonical_key"]]
            matched_ex[i] = j
            matched_inc[j] = i

    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    unmatched_inc = [j for j in range(len(inc_items)) if j not in matched_inc]

    if unmatched_ex and unmatched_inc:
        ex_texts = [
            f"{ex_items[i]['institution']} {ex_items[i].get('degree') or ''}"
            for i in unmatched_ex
        ]
        inc_texts = [
            f"{inc_items[j]['institution']} {inc_items[j].get('degree') or ''}"
            for j in unmatched_inc
        ]
        sim = _sim_matrix(
            _embed_texts(ex_texts, model_name),
            _embed_texts(inc_texts, model_name),
        )
        for i_local, j_local, _ in _greedy_pairs(sim, threshold):
            i_global = unmatched_ex[i_local]
            j_global = unmatched_inc[j_local]
            matched_ex[i_global] = j_global
            matched_inc[j_global] = i_global

    merged_records: List[Dict] = []
    result: List[Dict] = []
    processed_inc: set = set()

    for i, ex in enumerate(ex_items):
        if i in matched_ex:
            j = matched_ex[i]
            inc = inc_items[j]
            merged = {
                "canonical_key": ex["canonical_key"],
                "institution": _richer_str(ex["institution"], inc["institution"]) or ex["institution"],
                "degree": ex.get("degree") or inc.get("degree"),
                "field_of_study": _richer_str(ex.get("field_of_study"), inc.get("field_of_study")),
                "date_range": ex.get("date_range") or inc.get("date_range"),
                "gpa": ex.get("gpa") or inc.get("gpa"),
                "sources": _merge_sources(ex.get("sources", []), inc.get("sources", [])),
                "confidence": max(ex.get("confidence", 1.0), inc.get("confidence", 1.0)),
            }
            merged_records.append({"existing": ex, "incoming": inc, "result": merged})
            result.append(merged)
            processed_inc.add(j)
        else:
            result.append(ex)

    added = []
    for j, inc in enumerate(inc_items):
        if j not in processed_inc:
            result.append(inc)
            added.append(inc)

    return result, added, merged_records


def _merge_cert_lists(
    existing: List[Any],
    incoming: List[Any],
    src_ex: str,
    src_inc: str,
    threshold: float,
    model_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    def _build(name: str, source: str, existing_data: Any = None) -> Dict:
        if isinstance(existing_data, dict) and "canonical_key" in existing_data:
            d = dict(existing_data)
            sources = d.get("sources", [])
            if source not in sources:
                d["sources"] = sources + [source]
            return d
        return {
            "canonical_key": _canonical_key(name),
            "normalized_name": name,
            "aliases": [name],
            "sources": [source],
            "confidence": 1.0,
        }

    ex_items = [_build(_coerce_cert(x) or "", src_ex, x) for x in existing if _coerce_cert(x)]
    inc_items = [_build(_coerce_cert(x) or "", src_inc, x) for x in incoming if _coerce_cert(x)]
    ex_items = [x for x in ex_items if x.get("canonical_key")]
    inc_items = [x for x in inc_items if x.get("canonical_key")]

    ex_by_key = {c["canonical_key"]: i for i, c in enumerate(ex_items)}
    matched_ex: Dict[int, int] = {}
    matched_inc: Dict[int, int] = {}

    for j, inc in enumerate(inc_items):
        if inc["canonical_key"] in ex_by_key:
            i = ex_by_key[inc["canonical_key"]]
            matched_ex[i] = j
            matched_inc[j] = i

    unmatched_ex = [i for i in range(len(ex_items)) if i not in matched_ex]
    unmatched_inc = [j for j in range(len(inc_items)) if j not in matched_inc]

    if unmatched_ex and unmatched_inc:
        ex_texts = [ex_items[i]["normalized_name"] for i in unmatched_ex]
        inc_texts = [inc_items[j]["normalized_name"] for j in unmatched_inc]
        sim = _sim_matrix(
            _embed_texts(ex_texts, model_name),
            _embed_texts(inc_texts, model_name),
        )
        for i_local, j_local, _ in _greedy_pairs(sim, threshold):
            i_global = unmatched_ex[i_local]
            j_global = unmatched_inc[j_local]
            matched_ex[i_global] = j_global
            matched_inc[j_global] = i_global

    merged_records: List[Dict] = []
    result: List[Dict] = []
    processed_inc: set = set()

    for i, ex in enumerate(ex_items):
        if i in matched_ex:
            j = matched_ex[i]
            inc = inc_items[j]
            merged = {
                "canonical_key": ex["canonical_key"],
                "normalized_name": _richer_str(ex["normalized_name"], inc["normalized_name"]) or ex["normalized_name"],
                "aliases": list(dict.fromkeys(ex.get("aliases", []) + inc.get("aliases", []))),
                "sources": _merge_sources(ex.get("sources", []), inc.get("sources", [])),
                "confidence": max(ex.get("confidence", 1.0), inc.get("confidence", 1.0)),
            }
            merged_records.append({"existing": ex, "incoming": inc, "result": merged})
            result.append(merged)
            processed_inc.add(j)
        else:
            result.append(ex)

    added = []
    for j, inc in enumerate(inc_items):
        if j not in processed_inc:
            result.append(inc)
            added.append(inc)

    return result, added, merged_records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_profiles(
    existing_profile: Dict[str, Any],
    incoming_profile: Dict[str, Any],
    threshold: float = 0.85,
    model_name: str = "BAAI/bge-small-en",
) -> Dict[str, Any]:
    """
    Merge two profile dicts into a canonical deduplicated CanonicalProfile dict.

    Accepts both raw ExtractResponse shape and CanonicalProfile shape for each
    input.  Works field-by-field:
      1. Normalize skill names via alias map
      2. Exact canonical_key dedup
      3. Embedding similarity (threshold) for remaining
      4. Richer-wins heuristic: longer string, higher proficiency rank, union lists
      5. Conflicts surfaced (not blocking) for experience date mismatches

    Args:
        existing_profile: The current canonical or raw profile.
        incoming_profile: A new profile to merge in.
        threshold: Cosine similarity threshold for fuzzy dedup (0-1).
        model_name: SentenceTransformer model for embedding comparisons.

    Returns:
        Dict with keys: canonical_profile, added_items, merged_duplicates,
        conflicts, stats.

    Raises:
        ValueError: if both profiles are completely empty.
    """
    src_ex = _source_from_profile(existing_profile)
    src_inc = _source_from_profile(incoming_profile)

    ex_skills = list(existing_profile.get("skills") or [])
    ex_projects = list(existing_profile.get("projects") or [])
    ex_experience = list(existing_profile.get("experience") or [])
    ex_education = list(existing_profile.get("education") or [])
    ex_certs = list(existing_profile.get("certifications") or [])
    ex_keywords = [str(k) for k in (existing_profile.get("keywords") or [])]

    inc_skills = list(incoming_profile.get("skills") or [])
    inc_projects = list(incoming_profile.get("projects") or [])
    inc_experience = list(incoming_profile.get("experience") or [])
    inc_education = list(incoming_profile.get("education") or [])
    inc_certs = list(incoming_profile.get("certifications") or [])
    inc_keywords = [str(k) for k in (incoming_profile.get("keywords") or [])]

    logger.info(
        "merge_profiles src_ex=%s src_inc=%s threshold=%.2f",
        src_ex, src_inc, threshold,
    )

    merged_skills, added_skills, dup_skills = _merge_skill_lists(
        ex_skills, inc_skills, src_ex, src_inc, threshold, model_name
    )
    merged_projects, added_projects, dup_projects = _merge_project_lists(
        ex_projects, inc_projects, src_ex, src_inc, threshold, model_name
    )
    merged_exp, added_exp, dup_exp, exp_conflicts = _merge_experience_lists(
        ex_experience, inc_experience, src_ex, src_inc, threshold, model_name
    )
    merged_edu, added_edu, dup_edu = _merge_education_lists(
        ex_education, inc_education, src_ex, src_inc, threshold, model_name
    )
    merged_certs, added_certs, dup_certs = _merge_cert_lists(
        ex_certs, inc_certs, src_ex, src_inc, threshold, model_name
    )

    merged_keywords = _union_list(ex_keywords, inc_keywords)

    all_sources = _merge_sources(
        _sources_from_profile(existing_profile),
        _sources_from_profile(incoming_profile),
    )
    canonical_ids = [c["canonical_key"] for c in merged_skills + merged_projects]

    canonical_profile = {
        "skills": merged_skills,
        "projects": merged_projects,
        "experience": merged_exp,
        "education": merged_edu,
        "certifications": merged_certs,
        "keywords": merged_keywords,
        "source_documents": all_sources,
        "canonical_ids": canonical_ids,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "canonical_profile": canonical_profile,
        "added_items": {
            "skills": added_skills,
            "projects": added_projects,
            "experience": added_exp,
            "education": added_edu,
            "certifications": added_certs,
        },
        "merged_duplicates": {
            "skills": dup_skills,
            "projects": dup_projects,
            "experience": dup_exp,
            "education": dup_edu,
            "certifications": dup_certs,
        },
        "conflicts": {
            "experience": exp_conflicts,
        },
        "stats": {
            "skills_before": len(ex_skills) + len(inc_skills),
            "skills_after": len(merged_skills),
            "projects_before": len(ex_projects) + len(inc_projects),
            "projects_after": len(merged_projects),
            "experience_before": len(ex_experience) + len(inc_experience),
            "experience_after": len(merged_exp),
            "education_before": len(ex_education) + len(inc_education),
            "education_after": len(merged_edu),
            "certifications_before": len(ex_certs) + len(inc_certs),
            "certifications_after": len(merged_certs),
        },
    }
