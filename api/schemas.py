"""
API-layer request/response models (Step 19).

Deliberately thin wrappers around src/schemas.py's pydantic models rather
than a parallel data model -- the API should expose exactly what the
pipeline produces, not a reinterpretation of it. This keeps the FastAPI
layer and the Streamlit UI backed by the same contract.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.schemas import MatchExplanation, ScoreBreakdown, SkillComparisonItem, SkillGapItem


class AnalyzeJobDescriptionText(BaseModel):
    """Used when the job description is pasted as text rather than
    uploaded as a file (matches the Streamlit UI's "Paste text" mode).
    """
    text: str


class AnalyzeResponse(BaseModel):
    resume_filename: Optional[str]
    resume_name: Optional[str]
    job_title: Optional[str]
    score_breakdown: ScoreBreakdown
    skill_comparison: List[SkillComparisonItem]
    skill_gaps: List[SkillGapItem]
    explanation: MatchExplanation
    baseline_tfidf_similarity: Optional[float]
    resume_extraction_warning: Optional[str] = None
    job_extraction_warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    taxonomy_skill_count: int
    embeddings_available: bool
    mlflow_available: bool


class ErrorResponse(BaseModel):
    detail: str
