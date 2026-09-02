"""
Internal data contracts (Section 6 of the spec).

These are the canonical shapes passed between pipeline stages:
ingestion -> section parsing -> skill extraction -> matching -> scoring ->
explainability -> recommendations -> report.

Built with Pydantic so every stage gets validation "for free" and the
FastAPI layer can reuse these same models for its response schemas.

NOTE: this module requires `pydantic` (see requirements.txt). It is not
importable in this sandboxed dev environment (no network to install
pydantic), but is syntactically complete and ready to run in the target
environment.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src._pydantic_compat import BaseModel, Field


class SkillEvidence(BaseModel):
    """A single detected skill plus the evidence that supports it.

    Storing the evidence snippet is what lets the system say "AWS
    exposure was mentioned in this sentence" instead of silently
    upgrading a passing mention into an unqualified claim of expertise.
    """

    skill: str  # canonical/normalized name, e.g. "AWS"
    raw_mentions: List[str] = Field(default_factory=list)  # surface forms found
    evidence_snippets: List[str] = Field(default_factory=list)
    source_section: Optional[str] = None  # e.g. "experience", "skills", "projects"
    confidence: float = 1.0  # 0-1; lower for weak/contextual mentions


class ExperienceEntry(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    bullets: List[str] = Field(default_factory=list)
    raw_text: str = ""


class EducationEntry(BaseModel):
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    graduation_date: Optional[str] = None
    raw_text: str = ""


class ProjectEntry(BaseModel):
    name: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    raw_text: str = ""


class ResumeDocument(BaseModel):
    raw_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    skills: List[SkillEvidence] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)

    # canonical_section_name -> raw text block, e.g. {"experience": "..."}
    sections: Dict[str, str] = Field(default_factory=dict)

    # derived, filled in later in the pipeline
    total_years_experience: Optional[float] = None

    source_filename: Optional[str] = None
    extraction_warning: Optional[str] = None


class JobDocument(BaseModel):
    raw_text: str
    title: Optional[str] = None
    company: Optional[str] = None
    summary: Optional[str] = None

    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)

    # e.g. {"min_years": 3, "seniority": "senior", "raw": "3+ years..."}
    experience_requirements: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None

    education_requirements: List[str] = Field(default_factory=list)

    sections: Dict[str, str] = Field(default_factory=dict)

    source_filename: Optional[str] = None
    extraction_warning: Optional[str] = None


class SkillComparisonItem(BaseModel):
    skill: str
    status: str  # "MATCHED" | "PARTIAL" | "MISSING"
    requirement_level: str  # "required" | "preferred"
    evidence_snippets: List[str] = Field(default_factory=list)
    related_skill_found: Optional[str] = None  # if PARTIAL via a related skill


class SkillGapItem(BaseModel):
    skill: str
    requirement_level: str  # "required" | "preferred"
    priority: str  # "HIGH" | "MEDIUM" | "LOW"
    reason: str


class MatchExplanation(BaseModel):
    positive_evidence: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    semantic_evidence: List[dict] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    skills_score: float
    experience_score: float
    semantic_score: float
    projects_score: float
    education_score: float
    final_score: float
    weights_used: Dict[str, float]
    semantic_source: str = "tfidf_fallback"  # "embedding" | "tfidf_fallback"


class MatchResult(BaseModel):
    resume: ResumeDocument
    job: JobDocument
    score_breakdown: ScoreBreakdown
    skill_comparison: List[SkillComparisonItem] = Field(default_factory=list)
    skill_gaps: List[SkillGapItem] = Field(default_factory=list)
    explanation: MatchExplanation
    baseline_tfidf_similarity: Optional[float] = None
    semantic_similarity: Optional[float] = None
