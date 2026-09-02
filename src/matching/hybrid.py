"""
Hybrid matching engine (Section 14, Step 10).

Combines five signals into a single "hybrid similarity" score:
    lexical_similarity      (TF-IDF baseline)      weight 0.20
    semantic_similarity     (embeddings)            weight 0.30
    required_skill_match    (coverage ratio)        weight 0.30
    preferred_skill_match   (coverage ratio)        weight 0.10
    section_relevance       (experience section     weight 0.10
                             semantic match, proxy
                             for "relevant experience")

This is a RESEARCH/COMPARISON-LAYER score (used in the Model Lab to show
"here's what a hybrid approach would produce"), distinct from the
product-facing Final Match Score in src/scoring/scoring_engine.py, which
uses a different, more interpretable feature set (skills/experience/
semantic/projects/education). Keeping them separate mirrors the spec's own
split between Section 14 (hybrid architecture) and Section 16 (final
scoring) and avoids conflating "what predicts fit best" with "what's most
explainable to a candidate".

Graceful degradation: when semantic similarity is unavailable, its weight
is proportionally redistributed across the remaining components rather
than silently treating a missing signal as zero (which would understate
the score for a reason that has nothing to do with the candidate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.matching.semantic import compute_semantic_match
from src.matching.skill_matching import classify_skills, coverage_ratios
from src.matching.tfidf import tfidf_similarity
from src.nlp.skill_extraction import SkillTaxonomy
from src.schemas import JobDocument, ResumeDocument
from src.utils.config import CONFIG
from src.utils.logging import get_logger

logger = get_logger("matching.hybrid")


@dataclass
class HybridMatchResult:
    hybrid_score: float
    components: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    semantic_available: bool = False
    semantic_unavailable_reason: Optional[str] = None


def compute_hybrid_match(resume: ResumeDocument, job: JobDocument, taxonomy: SkillTaxonomy) -> HybridMatchResult:
    weights = dict(CONFIG.hybrid_weights.as_dict())

    lexical = tfidf_similarity(resume.raw_text, job.raw_text).similarity

    semantic_result = compute_semantic_match(resume, job)
    semantic_available = semantic_result.available
    semantic_value = semantic_result.overall_similarity

    candidate_skills = {s.skill: s for s in resume.skills}
    comparisons = classify_skills(candidate_skills, job.required_skills, job.preferred_skills, taxonomy)
    coverage = coverage_ratios(comparisons)

    experience_section_result = next(
        (r for r in semantic_result.section_results if r.section == "experience"), None
    )
    section_relevance = (
        experience_section_result.similarity
        if experience_section_result and experience_section_result.available
        else None
    )

    components = {
        "lexical_similarity": lexical,
        "semantic_similarity": semantic_value if semantic_value is not None else 0.0,
        "required_skill_match": coverage["required_coverage"],
        "preferred_skill_match": coverage["preferred_coverage"],
        "section_relevance": section_relevance if section_relevance is not None else lexical,
    }

    # Redistribute the semantic_similarity weight if it's unavailable,
    # rather than letting a missing signal silently drag the score down.
    if not semantic_available:
        missing_weight = weights["semantic_similarity"]
        weights["semantic_similarity"] = 0.0
        remaining_keys = [k for k in weights if k != "semantic_similarity"]
        remaining_total = sum(weights[k] for k in remaining_keys)
        for k in remaining_keys:
            weights[k] += missing_weight * (weights[k] / remaining_total)

    hybrid_score = sum(components[k] * weights[k] for k in components)

    return HybridMatchResult(
        hybrid_score=round(hybrid_score, 4),
        components={k: round(v, 4) for k, v in components.items()},
        weights_used={k: round(v, 4) for k, v in weights.items()},
        semantic_available=semantic_available,
        semantic_unavailable_reason=semantic_result.unavailable_reason,
    )
