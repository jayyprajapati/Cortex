from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1: Extraction schemas
# ---------------------------------------------------------------------------

class ResumeSkill(BaseModel):
    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None


class ResumeProject(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    date_range: Optional[str] = None


class ResumeExperience(BaseModel):
    company: str
    title: str
    date_range: Optional[str] = None
    location: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class ResumeEducation(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    date_range: Optional[str] = None
    gpa: Optional[str] = None


class ExtractMetadata(BaseModel):
    source_type: str
    parsed_at: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractRequest(BaseModel):
    app_name: str
    user_id: str
    doc_id: Optional[str] = None
    file_path: Optional[str] = None
    text: Optional[str] = None
    extraction_type: Literal["resume", "generic_profile", "structured_doc"] = "resume"


class ExtractResponse(BaseModel):
    doc_id: str
    document_type: str
    skills: List[Any] = Field(default_factory=list)
    projects: List[Any] = Field(default_factory=list)
    experience: List[Any] = Field(default_factory=list)
    education: List[Any] = Field(default_factory=list)
    certifications: List[Any] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    raw_sections: Dict[str, Any] = Field(default_factory=dict)
    metadata: ExtractMetadata


# ---------------------------------------------------------------------------
# Phase 2: Canonical profile schemas
# ---------------------------------------------------------------------------

class CanonicalSkill(BaseModel):
    """A deduplicated, normalized skill entry spanning one or more source docs."""
    canonical_key: str
    normalized_name: str
    aliases: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    proficiency: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CanonicalProject(BaseModel):
    """A deduplicated project, with merged technologies from all occurrences."""
    canonical_key: str
    normalized_name: str
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    date_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CanonicalExperience(BaseModel):
    """A deduplicated experience entry with semantically merged bullets."""
    canonical_key: str
    company: str
    title: str
    date_range: Optional[str] = None
    location: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CanonicalEducation(BaseModel):
    """A deduplicated education entry."""
    canonical_key: str
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    date_range: Optional[str] = None
    gpa: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CanonicalCertification(BaseModel):
    """A deduplicated certification with all observed name forms."""
    canonical_key: str
    normalized_name: str
    aliases: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CanonicalProfile(BaseModel):
    """
    The result of merging one or more ExtractResponse dicts.
    Deduplicated across all input documents with source traceability.
    """
    skills: List[CanonicalSkill] = Field(default_factory=list)
    projects: List[CanonicalProject] = Field(default_factory=list)
    experience: List[CanonicalExperience] = Field(default_factory=list)
    education: List[CanonicalEducation] = Field(default_factory=list)
    certifications: List[CanonicalCertification] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    canonical_ids: List[str] = Field(default_factory=list)
    merged_at: str


class ProfileMergeRequest(BaseModel):
    app_name: str
    user_id: str
    existing_profile: Dict[str, Any]
    incoming_profile: Dict[str, Any]
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class ProfileMergeResponse(BaseModel):
    canonical_profile: CanonicalProfile
    added_items: Dict[str, List[Any]]
    merged_duplicates: Dict[str, List[Any]]
    conflicts: Dict[str, List[Any]]
    stats: Dict[str, int]


# ---------------------------------------------------------------------------
# Phase 3: JD match + document generation schemas
# ---------------------------------------------------------------------------

class SectionRewrites(BaseModel):
    """Suggested rewrites for specific resume sections, keyed by section name."""
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[Any] = Field(default_factory=list)


class MatchRequest(BaseModel):
    """
    Input for POST /analyze/match.
    Compares a job description against the user's canonical profile and
    optionally their current submitted resume.
    """
    app_name: str
    user_id: str
    job_description: str
    canonical_profile: Dict[str, Any]
    base_resume: Optional[Dict[str, Any]] = None


class MatchResponse(BaseModel):
    """
    Structured JD-vs-profile analysis result.

    The critical distinction:
      missing_keywords                 — keywords the candidate does NOT have at all
      existing_but_missing_from_resume — keywords in the canonical profile that
                                         were omitted from the current resume
    """
    match_score: float = Field(ge=0.0, le=100.0)
    required_keywords: List[str]
    missing_keywords: List[str]
    existing_but_missing_from_resume: List[str]
    irrelevant_content: List[str]
    recommended_additions: List[str]
    recommended_removals: List[str]
    section_rewrites: Dict[str, Any]
    ats_keyword_clusters: Dict[str, List[str]]
    role_seniority: str
    domain_fit: str


class ExperienceBlock(BaseModel):
    """A single experience entry in the generated document."""
    company: str
    title: str
    date_range: Optional[str] = None
    location: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    """
    Input for POST /generate/document.
    Generates structured, ATS-optimized resume content blocks
    from the canonical profile, targeted at a specific JD and template type.
    """
    app_name: str
    user_id: str
    job_description: str
    canonical_profile: Dict[str, Any]
    base_resume: Optional[Dict[str, Any]] = None
    template_type: Literal["frontend", "backend", "fullstack"] = "fullstack"


class DocumentResponse(BaseModel):
    """
    Structured resume content blocks ready for template injection.
    Contains only truthful content drawn from the canonical profile.
    """
    summary: str
    skills: List[str]
    projects: List[Any]
    experience: List[Any]
    target_keywords_used: List[str]
    removed_content: List[str]
    match_score_improved: float = Field(ge=0.0, le=100.0)
