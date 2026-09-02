"""
High-level orchestration: raw bytes in, full match result out.

Both the Streamlit app and the FastAPI service should call THIS module
rather than wiring together ingestion/parsing/scoring themselves --
per the spec's instruction to keep business logic out of UI files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.ingestion.document_ingestion import IngestionResult, ingest_document
from src.matching.semantic import compute_semantic_match
from src.nlp.job_parser import parse_job_description
from src.nlp.resume_parser import parse_resume
from src.nlp.resume_quality import ResumeQualityReport, analyze_resume_quality
from src.nlp.skill_extraction import SkillTaxonomy
from src.recommendations.recommendation_engine import Recommendation, build_recommendations
from src.recommendations.roadmap import SkillRoadmap, build_roadmap
from src.schemas import JobDocument, ResumeDocument
from src.scoring.scoring_engine import run_full_match
from src.utils.logging import get_logger

logger = get_logger("pipeline")

# Loaded once per process; SkillTaxonomy just parses a small JSON file, but
# Streamlit callers should still wrap this with @st.cache_resource.
_TAXONOMY: Optional[SkillTaxonomy] = None


def get_taxonomy() -> SkillTaxonomy:
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = SkillTaxonomy()
    return _TAXONOMY


@dataclass
class AnalysisResult:
    resume: ResumeDocument
    job: JobDocument
    match: Dict
    resume_ingestion: IngestionResult
    job_ingestion: Optional[IngestionResult]
    quality: ResumeQualityReport
    recommendations: list[Recommendation]
    roadmap: SkillRoadmap


def analyze_resume_vs_job(
    resume_bytes: bytes,
    resume_filename: str,
    job_text_or_bytes,
    job_filename: Optional[str] = None,
) -> AnalysisResult:
    """Run the full pipeline for one resume + one job description.

    `job_text_or_bytes` may be raw pasted text (str) or an uploaded file's
    bytes (if job_filename is also given).
    """
    taxonomy = get_taxonomy()

    resume_ingestion = ingest_document(resume_bytes, resume_filename)
    resume = parse_resume(
        resume_ingestion.raw_text,
        taxonomy,
        source_filename=resume_filename,
        extraction_warning=resume_ingestion.quality.message,
    )

    job_ingestion: Optional[IngestionResult] = None
    if job_filename:
        job_ingestion = ingest_document(job_text_or_bytes, job_filename)
        job_raw_text = job_ingestion.raw_text
        job_warning = job_ingestion.quality.message
    else:
        job_raw_text = job_text_or_bytes
        job_warning = None

    job = parse_job_description(job_raw_text, taxonomy, source_filename=job_filename)
    job.extraction_warning = job_warning

    # Compute real semantic similarity from embeddings when available;
    # compute_semantic_match itself reports .available=False (with a
    # reason) rather than throwing when the model can't be loaded, so
    # this degrades to the documented TF-IDF fallback inside score_match
    # instead of crashing the whole analysis.
    semantic_result = compute_semantic_match(resume, job)
    semantic_similarity = semantic_result.overall_similarity if semantic_result.available else None

    match = run_full_match(resume, job, taxonomy, semantic_similarity=semantic_similarity)
    quality = analyze_resume_quality(resume)
    recommendations = build_recommendations(match["skill_gaps"], quality)
    roadmap = build_roadmap(match["skill_gaps"])

    return AnalysisResult(
        resume=resume,
        job=job,
        match=match,
        resume_ingestion=resume_ingestion,
        job_ingestion=job_ingestion,
        quality=quality,
        recommendations=recommendations,
        roadmap=roadmap,
    )
