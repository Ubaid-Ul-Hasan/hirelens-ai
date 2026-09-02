"""
Final match scoring (Section 16) and feature engineering (Section 15).

Deliberately keeps the "Semantic Fit" component optional/pluggable:
sentence-transformers isn't available in every environment (e.g. this
dev sandbox, or a minimal deployment), and per Engineering Rule #19
("keep the application usable even if advanced components fail"), the
whole pipeline must still produce a usable score using only TF-IDF +
structured features when semantic similarity can't be computed -- with
that degradation clearly recorded, never silently hidden.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.matching.skill_matching import classify_skills, coverage_ratios
from src.matching.tfidf import tfidf_similarity
from src.nlp.skill_extraction import SkillTaxonomy
from src.recommendations.skill_gap import build_skill_gaps
from src.schemas import JobDocument, MatchExplanation, ResumeDocument, ScoreBreakdown, SkillComparisonItem
from src.utils.config import CONFIG
from src.utils.logging import get_logger

logger = get_logger("scoring.engine")


def _experience_score(resume: ResumeDocument, job: JobDocument) -> float:
    """Compare candidate's stated years of experience against the job's
    minimum, if both are known. Returns 1.0 (full credit) when the job
    states no minimum, and 0.5 (neutral, not punitive) when the candidate's
    years simply weren't detected -- absence of a detected number is not
    evidence the candidate lacks experience (Engineering Rule #14).
    """
    if job.min_years_experience is None:
        return 1.0
    if resume.total_years_experience is None:
        return 0.5
    if resume.total_years_experience >= job.min_years_experience:
        return 1.0
    # partial credit scaled by how close they are, floor at 0
    ratio = resume.total_years_experience / job.min_years_experience
    return max(0.0, min(1.0, ratio))


def _projects_score(resume: ResumeDocument, comparisons: List[SkillComparisonItem]) -> float:
    """Rough proxy: does the candidate have any projects at all, and do
    project bullets reference any of the job's required/matched skills?
    This is intentionally simple in the MVP -- a fuller implementation
    would run skill extraction over project text specifically and compare
    against job requirements directly (tracked as a documented extension).
    """
    if not resume.projects:
        return 0.3  # not zero: absence of a "Projects" section isn't damning
    matched_skills = {c.skill.lower() for c in comparisons if c.status == "MATCHED"}
    project_text = " ".join(p.raw_text.lower() for p in resume.projects)
    hits = sum(1 for skill in matched_skills if skill in project_text)
    if not matched_skills:
        return 0.6
    return min(1.0, 0.4 + 0.6 * (hits / max(1, len(matched_skills))))


def _education_score(resume: ResumeDocument, job: JobDocument) -> float:
    """If the job states no education requirement, full credit. Otherwise,
    full credit if the candidate has ANY education entry (a stricter
    degree-level match is a documented future extension -- see spec
    Section 9 on not over-interpreting loosely-stated requirements).
    """
    if not job.education_requirements:
        return 1.0
    return 1.0 if resume.education else 0.4


def score_match(
    resume: ResumeDocument,
    job: JobDocument,
    taxonomy: SkillTaxonomy,
    semantic_similarity: Optional[float] = None,
) -> ScoreBreakdown:
    candidate_skills = {s.skill: s for s in resume.skills}

    comparisons = classify_skills(candidate_skills, job.required_skills, job.preferred_skills, taxonomy)
    coverage = coverage_ratios(comparisons)

    skills_score = (
        0.7 * coverage["required_coverage"] + 0.3 * coverage["preferred_coverage"]
        if (job.required_skills or job.preferred_skills)
        else 1.0
    )

    experience_score = _experience_score(resume, job)
    projects_score = _projects_score(resume, comparisons)
    education_score = _education_score(resume, job)

    baseline = tfidf_similarity(resume.raw_text, job.raw_text)

    if semantic_similarity is not None:
        semantic_score = semantic_similarity
        semantic_source = "embedding"
    else:
        # graceful degradation: fall back to the TF-IDF baseline as the
        # semantic-fit proxy, and say so explicitly rather than hiding it
        semantic_score = baseline.similarity
        semantic_source = "tfidf_fallback"
        logger.info("Semantic similarity unavailable; falling back to TF-IDF baseline as proxy.")

    weights = CONFIG.final_weights
    final_score = (
        weights.skills * skills_score
        + weights.experience * experience_score
        + weights.semantic_fit * semantic_score
        + weights.projects * projects_score
        + weights.education * education_score
    )

    breakdown = ScoreBreakdown(
        skills_score=round(skills_score, 4),
        experience_score=round(experience_score, 4),
        semantic_score=round(semantic_score, 4),
        projects_score=round(projects_score, 4),
        education_score=round(education_score, 4),
        final_score=round(final_score, 4),
        weights_used=weights.as_dict(),
        semantic_source=semantic_source,
    )
    return breakdown, comparisons, baseline


def build_explanation(
    resume: ResumeDocument, job: JobDocument, comparisons: List[SkillComparisonItem]
) -> MatchExplanation:
    positive = [
        f"{c.skill} detected" + (f" (evidence: \"{c.evidence_snippets[0][:80]}...\")" if c.evidence_snippets else "")
        for c in comparisons
        if c.status == "MATCHED"
    ]
    negative = []
    for c in comparisons:
        if c.status == "MISSING":
            negative.append(f"{c.skill} was not detected in the supplied resume.")
        elif c.status == "PARTIAL":
            negative.append(
                f"{c.skill} not detected directly; related skill {c.related_skill_found} was found instead."
            )

    if job.min_years_experience is not None and resume.total_years_experience is not None:
        if resume.total_years_experience < job.min_years_experience:
            negative.append(
                f"Required experience ({job.min_years_experience:.0f}+ years) exceeds "
                f"detected experience ({resume.total_years_experience:.0f} years)."
            )

    return MatchExplanation(positive_evidence=positive, negative_evidence=negative, semantic_evidence=[])


def run_full_match(
    resume: ResumeDocument,
    job: JobDocument,
    taxonomy: SkillTaxonomy,
    semantic_similarity: Optional[float] = None,
) -> Dict:
    """End-to-end: score, explain, and compute gaps for one resume/job pair.
    Returns a plain dict (not a pydantic MatchResult) to stay usable even
    in environments without pydantic installed (see src/_pydantic_compat).
    """
    breakdown, comparisons, baseline = score_match(resume, job, taxonomy, semantic_similarity)
    gaps = build_skill_gaps(comparisons)
    explanation = build_explanation(resume, job, comparisons)

    return {
        "score_breakdown": breakdown,
        "skill_comparison": comparisons,
        "skill_gaps": gaps,
        "explanation": explanation,
        "baseline_tfidf_similarity": baseline.similarity,
        "baseline_vocabulary_size": baseline.vocabulary_size,
    }
