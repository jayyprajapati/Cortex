"""
resume_optimizer.py

JD-vs-profile analysis and ATS-optimized document generation.

Two public entry points:
  analyze_match()     — Score a JD against a canonical profile; identify
                        missing skills (A: user lacks entirely) vs. omitted
                        skills (B: user has in profile but not in current resume).

  generate_document() — Produce structured, truthful resume content blocks
                        optimized for a target JD and template type.

Both entry points:
  1. Pre-compute deterministic keyword sets (no LLM needed for the A/B split)
  2. Format a compact context string for the LLM
  3. Call the LLM via the existing retry contract
  4. Validate and return a structured dict

No Qdrant reads or writes.  No fabrication.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.context import EffectiveGenerationConfig, LLMConfig
from app.llm.factory import get_llm
from app.pipeline.generate_pipeline import (
    _extract_json_payload,
    _normalize_schema,
    _validate_schema,
)

logger = logging.getLogger(__name__)

_LLM_AUTH_KEYWORDS = frozenset({
    "unauthorized", "forbidden", "authentication", "api key",
    "invalid token", "invalid key", " 401", " 403",
})


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).strip().lower()
    return any(kw in msg for kw in _LLM_AUTH_KEYWORDS)


# ---------------------------------------------------------------------------
# Template-type guidance injected into the generate prompt
# ---------------------------------------------------------------------------

_TEMPLATE_GUIDANCE: Dict[str, str] = {
    "frontend": (
        "Emphasize: UI frameworks (React, Vue, Angular), CSS/Tailwind, performance "
        "optimisation, accessibility (a11y), TypeScript, state management, "
        "bundlers (Webpack/Vite), and any design-system work. "
        "Deprioritise deep backend infrastructure, ML pipelines, and DevOps minutiae."
    ),
    "backend": (
        "Emphasise: REST/GraphQL API design, databases (SQL and NoSQL), "
        "distributed systems, scalability patterns, message queues, "
        "cloud services (AWS/GCP/Azure), and server-side languages (Python, Go, Java). "
        "Deprioritise frontend styling and pixel-level UI work."
    ),
    "fullstack": (
        "Balance frontend and backend depth. Highlight system design, "
        "full-stack ownership, API contracts, and the ability to ship "
        "end-to-end features independently."
    ),
}

# ---------------------------------------------------------------------------
# Deterministic keyword helpers
# ---------------------------------------------------------------------------

def _extract_profile_skills(canonical_profile: Dict[str, Any]) -> Set[str]:
    """All skill names and aliases from the canonical profile, lower-cased."""
    names: Set[str] = set()
    for item in canonical_profile.get("skills") or []:
        if isinstance(item, dict):
            name = str(item.get("normalized_name") or item.get("name") or "").strip()
            if name:
                names.add(name.lower())
            for alias in item.get("aliases") or []:
                if alias:
                    names.add(str(alias).strip().lower())
        elif isinstance(item, str) and item.strip():
            names.add(item.strip().lower())
    # Also include technologies from projects and certifications
    for proj in canonical_profile.get("projects") or []:
        if isinstance(proj, dict):
            for tech in proj.get("technologies") or []:
                if tech:
                    names.add(str(tech).strip().lower())
    for cert in canonical_profile.get("certifications") or []:
        cert_name = (
            str(cert.get("normalized_name") or cert.get("name") or "")
            if isinstance(cert, dict) else str(cert)
        ).strip().lower()
        if cert_name:
            names.add(cert_name)
    return names


def _extract_resume_skills(base_resume: Optional[Dict[str, Any]]) -> Set[str]:
    """Skill names from the current base resume (subset of the canonical profile)."""
    if not base_resume:
        return set()
    names: Set[str] = set()
    for item in base_resume.get("skills") or []:
        skill_name = (
            str(item.get("normalized_name") or item.get("name") or "")
            if isinstance(item, dict) else str(item)
        ).strip().lower()
        if skill_name:
            names.add(skill_name)
    for kw in base_resume.get("keywords") or []:
        if kw:
            names.add(str(kw).strip().lower())
    return names


def _compute_keyword_sets(
    canonical_profile: Dict[str, Any],
    base_resume: Optional[Dict[str, Any]],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns (profile_skills, resume_skills, omitted_skills).
    omitted_skills = profile_skills - resume_skills  (optimization gold mine)
    """
    profile_skills = _extract_profile_skills(canonical_profile)
    resume_skills = _extract_resume_skills(base_resume)
    omitted = profile_skills - resume_skills
    return profile_skills, resume_skills, omitted


# ---------------------------------------------------------------------------
# Profile → text formatter
# ---------------------------------------------------------------------------

def _fmt_skills(canonical_profile: Dict[str, Any], max_items: int = 40) -> str:
    items = []
    for s in (canonical_profile.get("skills") or [])[:max_items]:
        if isinstance(s, dict):
            name = str(s.get("normalized_name") or s.get("name") or "").strip()
            prof = str(s.get("proficiency") or "").strip()
            items.append(f"{name} ({prof})" if prof else name)
        elif isinstance(s, str):
            items.append(s.strip())
    return ", ".join(items) if items else "None listed"


def _fmt_experience(canonical_profile: Dict[str, Any], max_bullets: int = 3) -> str:
    lines = []
    for i, exp in enumerate((canonical_profile.get("experience") or [])[:6], 1):
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company") or "").strip()
        title = str(exp.get("title") or "").strip()
        date = str(exp.get("date_range") or "").strip()
        header = f"{i}. {company} | {title}"
        if date:
            header += f" | {date}"
        lines.append(header)
        for b in (exp.get("bullets") or [])[:max_bullets]:
            if b:
                lines.append(f"   • {b.strip()}")
    return "\n".join(lines) if lines else "None listed"


def _fmt_projects(canonical_profile: Dict[str, Any]) -> str:
    lines = []
    for proj in (canonical_profile.get("projects") or [])[:6]:
        if not isinstance(proj, dict):
            continue
        name = str(proj.get("normalized_name") or proj.get("name") or "").strip()
        desc = str(proj.get("description") or "").strip()
        techs = ", ".join(str(t) for t in (proj.get("technologies") or [])[:8])
        url = str(proj.get("url") or "").strip()
        line = f"• {name}"
        if techs:
            line += f" [{techs}]"
        if desc:
            line += f" — {desc[:120]}"
        if url:
            line += f" ({url})"
        lines.append(line)
    return "\n".join(lines) if lines else "None listed"


def _fmt_certifications(canonical_profile: Dict[str, Any]) -> str:
    certs = []
    for c in (canonical_profile.get("certifications") or [])[:10]:
        name = (
            str(c.get("normalized_name") or c.get("name") or "")
            if isinstance(c, dict) else str(c)
        ).strip()
        if name:
            certs.append(f"• {name}")
    return "\n".join(certs) if certs else "None listed"


def _format_profile_context(
    canonical_profile: Dict[str, Any],
    profile_skills: Set[str],
    resume_skills: Set[str],
    omitted_skills: Set[str],
) -> str:
    """
    Compose the full LLM context block:
    profile sections + pre-computed keyword set analysis.
    """
    omitted_display = sorted(omitted_skills)[:30]
    resume_display = sorted(resume_skills)[:30]

    return (
        "=== CANONICAL PROFILE ===\n"
        f"SKILLS: {_fmt_skills(canonical_profile)}\n\n"
        f"EXPERIENCE:\n{_fmt_experience(canonical_profile)}\n\n"
        f"PROJECTS:\n{_fmt_projects(canonical_profile)}\n\n"
        f"CERTIFICATIONS:\n{_fmt_certifications(canonical_profile)}\n\n"
        "=== PRE-COMPUTED KEYWORD ANALYSIS ===\n"
        f"ALL profile skills ({len(profile_skills)} total): "
        f"{', '.join(sorted(profile_skills)[:50])}\n\n"
        f"Skills in current resume ({len(resume_skills)}): "
        f"{', '.join(resume_display) if resume_display else 'Not provided'}\n\n"
        f"Skills in profile but OMITTED from resume ({len(omitted_display)}): "
        f"{', '.join(omitted_display) if omitted_display else 'None — resume appears complete'}\n"
    )


def _format_base_resume(base_resume: Optional[Dict[str, Any]]) -> str:
    if not base_resume:
        return "Not provided."
    lines = []
    skills = base_resume.get("skills") or []
    if skills:
        skill_names = [
            str(s.get("normalized_name") or s.get("name") or s) if isinstance(s, dict) else str(s)
            for s in skills[:30]
        ]
        lines.append(f"Skills: {', '.join(skill_names)}")
    for exp in (base_resume.get("experience") or [])[:4]:
        if isinstance(exp, dict):
            lines.append(
                f"{exp.get('company', '')} | {exp.get('title', '')} "
                f"| {exp.get('date_range', '')}"
            )
    for proj in (base_resume.get("projects") or [])[:3]:
        if isinstance(proj, dict):
            lines.append(f"Project: {proj.get('name', proj.get('normalized_name', ''))}")
    return "\n".join(lines) if lines else "Empty resume provided."


# ---------------------------------------------------------------------------
# LLM call with retry contract
# ---------------------------------------------------------------------------

def _call_llm_structured(
    prompt: str,
    schema: Dict[str, Any],
    llm_config: LLMConfig,
    max_retries: int,
    temperature: float = 0.05,
) -> Dict[str, Any]:
    """
    Call the LLM, parse JSON, validate against schema, retry on failure.
    Raises ValueError after max_retries exhausted.
    """
    llm = get_llm(llm_config)
    normalized_schema = _normalize_schema(schema)
    retry_prompt = prompt
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(retry_prompt, temperature=temperature)
            parsed = _extract_json_payload(raw)
            if normalized_schema:
                _validate_schema(parsed, normalized_schema)
            logger.info(
                "optimizer LLM success attempt=%d/%d",
                attempt + 1, max_retries + 1,
            )
            return parsed
        except Exception as exc:
            if _is_auth_error(exc):
                raise ValueError(
                    f"LLM provider authentication failed: {exc}. "
                    "Verify your API key in the environment configuration."
                ) from exc
            last_error = exc
            logger.warning(
                "optimizer LLM attempt %d/%d failed: %s",
                attempt + 1, max_retries + 1, exc,
            )
            if attempt < max_retries:
                retry_prompt = (
                    f"{prompt}\n\n"
                    "CORRECTION: Your previous response was invalid JSON or did not "
                    "match the required schema. Return ONLY valid JSON. "
                    "No markdown fences, no prose outside the JSON object."
                )

    raise ValueError(
        f"LLM generation failed after {max_retries + 1} attempt(s). "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Match analysis prompts
# ---------------------------------------------------------------------------

_MATCH_PROMPT = """\
{system_prompt}

{context}

=== BASE RESUME (Current Submission) ===
{base_resume_text}

Rules:
- Return ONLY valid JSON. No markdown fences, no prose outside the JSON object.
- match_score: integer 0-100 based on keyword overlap, seniority fit, and domain alignment
- required_keywords: ALL skill/technology/methodology keywords found in the JD
- missing_keywords: required_keywords the candidate does NOT have anywhere in their profile
- existing_but_missing_from_resume: required_keywords present in the PROFILE but absent \
from the BASE RESUME — these are immediate optimization wins, add them to the resume
- irrelevant_content: items in the base resume that are low-signal or irrelevant for this JD
- recommended_additions: specific items from the canonical profile to ADD to resume (truthful only)
- recommended_removals: specific items in the base resume to REMOVE to improve signal/noise
- section_rewrites.summary: 2-3 sentence professional summary using JD keywords naturally
- section_rewrites.skills: ordered skill list, most JD-relevant first
- section_rewrites.projects: list of the most relevant project objects for this JD
- ats_keyword_clusters: group keywords by theme, e.g. {{"backend": [...], "cloud": [...]}}
- role_seniority: one of "junior", "mid", "senior", "lead", "principal", "executive"
- domain_fit: one concise sentence on how well the candidate's domain aligns with the role

Job Description:
{job_description}

JSON Output:"""


# ---------------------------------------------------------------------------
# Aggressiveness guidance
# ---------------------------------------------------------------------------

_AGGRESSIVENESS_GUIDANCE: Dict[str, str] = {
    "conservative": (
        "AGGRESSIVENESS: conservative — Minimal changes. Preserve the candidate's original "
        "phrasing and structure as much as possible. Only add clearly missing relevant "
        "keywords, fix glaring gaps, and remove obviously irrelevant one-liners. "
        "Do not rewrite bullets; at most, prepend or append a keyword naturally."
    ),
    "balanced": (
        "AGGRESSIVENESS: balanced — Moderate optimization. Reorder priorities to surface "
        "the most JD-relevant content first. Strengthen the top 2-3 bullets per role "
        "with JD keywords. Remove weak or low-relevance content. "
        "Rephrase lightly where the original wording buries important signal."
    ),
    "aggressive": (
        "AGGRESSIVENESS: aggressive — Maximum ATS optimization. Significantly restructure "
        "the resume for keyword density and relevance. Rewrite bullets for maximum impact "
        "using JD language. Remove all marginally relevant content. Bold keyword placement "
        "throughout. The structure may differ substantially from the source — "
        "but NEVER fabricate skills, metrics, companies, or titles not in the profile."
    ),
}


# ---------------------------------------------------------------------------
# Generate document prompts
# ---------------------------------------------------------------------------

_GENERATE_PROMPT = """\
{system_prompt}

{aggressiveness_guidance}

{context}

=== TEMPLATE TYPE: {template_type} ===
{template_guidance}

Rules:
- Return ONLY valid JSON. No markdown fences, no prose outside the JSON object.
- summary: 2-3 sentences, professional tone, naturally embed JD keywords
- skills: flat ordered list of skill name strings, most JD-relevant first, \
max 20 items; exclude irrelevant skills
- projects: array of project objects — pick the most JD-relevant projects from the \
canonical profile. Each object: {{name, description, technologies[string], url|null, \
date_range|null, relevance_note}}
- experience: array of experience objects — for each relevant role keep the 3-4 most \
JD-relevant bullets. Each object: {{company, title, date_range|null, location|null, \
bullets[string]}}
- target_keywords_used: flat list of JD keywords you successfully incorporated
- removed_content: flat list of plain strings ONLY — format each entry as \
"item name (reason)" where reason is one word: "irrelevant", "weak", or "redundant". \
Example: ["Ruby (irrelevant)", "Excel (weak)"]. NO objects, NO dicts — strings only.
- match_score_improved: your estimated ATS score for this generated document (0-100)
- TRUTHFUL ONLY: do NOT invent skills, metrics, companies, titles, dates, or \
project names not in the canonical profile
- Prefer historically relevant, high-impact experience over recent but low-relevance items
{extra_instructions}

Job Description:
{job_description}

JSON Output:"""


_MODIFY_PROMPT = """\
{system_prompt}

{aggressiveness_guidance}

You are modifying an existing resume to better target a specific job description.
Preserve the candidate's authentic voice. Perform targeted optimizations only.
NEVER fabricate experience, skills, metrics, companies, or titles.

=== SOURCE RESUME (preserve structure) ===
{source_resume_text}

{context}

=== TEMPLATE TYPE: {template_type} ===
{template_guidance}

Rules:
- Return ONLY valid JSON. No markdown fences, no prose outside the JSON object.
- summary: rewrite or lightly adjust the summary to feature JD keywords naturally
- skills: start from the source resume skills list; reorder and add profile skills \
  that match JD; remove irrelevant skills according to aggressiveness level
- projects: start from source resume projects; reorder by JD relevance; update \
  descriptions to surface relevant keywords (truthful only)
- experience: start from source resume experience; adjust bullets to front-load JD \
  keywords; reorder within each role by relevance; keep original company/title/dates
- target_keywords_used: flat list of JD keywords you successfully incorporated
- removed_content: flat list of plain strings ONLY — format each entry as \
  "item name (reason)" where reason is one word: "irrelevant", "weak", or "redundant". \
  Example: ["Ruby (irrelevant)", "Excel (weak)"]. NO objects, NO dicts — strings only.
- match_score_improved: your estimated ATS score for this modified document (0-100)
- TRUTHFUL ONLY: only rephrase what exists — never add invented content
{extra_instructions}

Job Description:
{job_description}

JSON Output:"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def analyze_match(
    job_description: str,
    canonical_profile: Dict[str, Any],
    base_resume: Optional[Dict[str, Any]],
    llm_config: LLMConfig,
    system_prompt: str,
    schema: Dict[str, Any],
    max_retries: int = 3,
    temperature: float = 0.05,
) -> Dict[str, Any]:
    """
    Score a job description against the canonical profile.

    Deterministic pre-computation identifies:
      A) Keywords the candidate lacks entirely → missing_keywords
      B) Keywords in canonical profile but omitted from base_resume
         → existing_but_missing_from_resume

    Args:
        job_description:   Raw JD text.
        canonical_profile: Phase 2 CanonicalProfile dict (or ExtractResponse shape).
        base_resume:       Optional current resume dict for gap detection.
        llm_config:        Resolved LLM provider config.
        system_prompt:     From registry task override.
        schema:            JSON Schema from registry task override.
        max_retries:       LLM retry count.
        temperature:       LLM temperature.

    Returns:
        Dict conforming to MatchResponse schema.

    Raises:
        ValueError: If JD is empty, profile has no extractable data, or LLM fails.
    """
    jd = str(job_description or "").strip()
    if not jd:
        raise ValueError("job_description cannot be empty")

    profile_skills, resume_skills, omitted = _compute_keyword_sets(canonical_profile, base_resume)

    context = _format_profile_context(canonical_profile, profile_skills, resume_skills, omitted)
    base_resume_text = _format_base_resume(base_resume)

    prompt = _MATCH_PROMPT.format(
        system_prompt=system_prompt,
        context=context,
        base_resume_text=base_resume_text,
        job_description=jd[:6000],
    )

    result = _call_llm_structured(prompt, schema, llm_config, max_retries, temperature)

    # Clamp match_score to 0-100
    if isinstance(result.get("match_score"), (int, float)):
        result["match_score"] = max(0.0, min(100.0, float(result["match_score"])))

    return result


def _format_source_resume(source_resume_content: Dict[str, Any]) -> str:
    """Format sectioned_resume_source into a readable prompt block."""
    lines = []
    section_order = ["summary", "skills", "experience", "projects", "education", "certifications", "contact"]
    seen = set()
    for section in section_order:
        if section in source_resume_content:
            val = source_resume_content[section]
            text = val if isinstance(val, str) else "\n".join(str(v) for v in val) if isinstance(val, list) else str(val)
            if text.strip():
                lines.append(f"[{section.upper()}]\n{text.strip()}")
                seen.add(section)
    for section, val in source_resume_content.items():
        if section not in seen:
            text = val if isinstance(val, str) else "\n".join(str(v) for v in val) if isinstance(val, list) else str(val)
            if text.strip():
                lines.append(f"[{section.upper()}]\n{text.strip()}")
    return "\n\n".join(lines) if lines else "No source resume content provided."


def generate_document(
    job_description: str,
    canonical_profile: Dict[str, Any],
    base_resume: Optional[Dict[str, Any]],
    template_type: str,
    llm_config: LLMConfig,
    system_prompt: str,
    schema: Dict[str, Any],
    max_retries: int = 3,
    temperature: float = 0.05,
    mode: str = "canonical_only",
    source_resume_content: Optional[Dict[str, Any]] = None,
    user_tweak_prompt: Optional[str] = None,
    include_missing_profile_keywords: bool = True,
    include_external_keywords: bool = False,
    remove_irrelevant_keywords: bool = True,
    aggressiveness: str = "balanced",
) -> Dict[str, Any]:
    """
    Generate structured, ATS-optimized resume content blocks.

    mode="canonical_only": Build fresh from canonical profile (default).
    mode="modify_existing": Targeted optimization of an existing resume structure.
    aggressiveness: "conservative" | "balanced" | "aggressive"
    """
    jd = str(job_description or "").strip()
    if not jd:
        raise ValueError("job_description cannot be empty")

    if mode == "modify_existing" and not source_resume_content:
        raise ValueError("modify_existing mode requires source_resume_content (sectioned_resume_source from /extract).")

    valid_types = {"frontend", "backend", "fullstack"}
    safe_template = template_type.strip().lower() if template_type else "fullstack"
    if safe_template not in valid_types:
        safe_template = "fullstack"

    safe_aggressiveness = aggressiveness if aggressiveness in _AGGRESSIVENESS_GUIDANCE else "balanced"
    aggressiveness_guidance = _AGGRESSIVENESS_GUIDANCE[safe_aggressiveness]
    template_guidance = _TEMPLATE_GUIDANCE[safe_template]

    profile_skills, resume_skills, omitted = _compute_keyword_sets(canonical_profile, base_resume)
    context = _format_profile_context(canonical_profile, profile_skills, resume_skills, omitted)

    # Build extra instructions from strategy flags
    extra_parts = []
    if user_tweak_prompt and str(user_tweak_prompt).strip():
        extra_parts.append(f"CANDIDATE PRIORITY INSTRUCTIONS: {str(user_tweak_prompt).strip()}")
    if include_missing_profile_keywords:
        extra_parts.append("- Include profile skills that are missing from the current resume but relevant to the JD.")
    if include_external_keywords and not include_missing_profile_keywords:
        extra_parts.append("- You may suggest adding industry-standard keywords not in the profile as 'suggested additions' — clearly mark them.")
    if remove_irrelevant_keywords:
        extra_parts.append("- Remove or deprioritize content with low relevance to this JD.")
    extra_instructions = "\n".join(extra_parts)

    if mode == "modify_existing":
        source_text = _format_source_resume(source_resume_content)
        prompt = _MODIFY_PROMPT.format(
            system_prompt=system_prompt,
            aggressiveness_guidance=aggressiveness_guidance,
            source_resume_text=source_text,
            context=context,
            template_type=safe_template.upper(),
            template_guidance=template_guidance,
            extra_instructions=extra_instructions,
            job_description=jd[:6000],
        )
    else:
        prompt = _GENERATE_PROMPT.format(
            system_prompt=system_prompt,
            aggressiveness_guidance=aggressiveness_guidance,
            context=context,
            template_type=safe_template.upper(),
            template_guidance=template_guidance,
            extra_instructions=extra_instructions,
            job_description=jd[:6000],
        )

    result = _call_llm_structured(prompt, schema, llm_config, max_retries, temperature)

    if isinstance(result.get("match_score_improved"), (int, float)):
        result["match_score_improved"] = max(0.0, min(100.0, float(result["match_score_improved"])))

    # Coerce removed_content entries to strings — the LLM sometimes returns dicts
    rc = result.get("removed_content")
    if isinstance(rc, list):
        coerced = []
        for item in rc:
            if isinstance(item, str):
                coerced.append(item)
            elif isinstance(item, dict):
                name = item.get("item") or item.get("name") or item.get("content") or ""
                reason = item.get("reason") or ""
                coerced.append(f"{name} ({reason})" if reason else name)
        result["removed_content"] = coerced

    result["mode"] = mode
    return result
