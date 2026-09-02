"""
Skill matching logic (Section 17).

Classifies each job-required/preferred skill against the candidate's
detected skills as MATCHED / PARTIAL / MISSING. A related-but-different
skill (e.g. candidate has TensorFlow, job wants PyTorch) is PARTIAL, never
an automatic MATCHED -- credit for it is configurable via
SkillMatchPolicy.allow_related_skill_partial_credit, and even when
enabled it never fully substitutes for the exact requirement.
"""
from __future__ import annotations

from typing import Dict, List

from src.nlp.skill_extraction import SkillTaxonomy
from src.schemas import SkillComparisonItem
from src.utils.config import CONFIG


def classify_skills(
    candidate_skills: Dict[str, "object"],  # canonical name -> SkillEvidence
    required_skills: List[str],
    preferred_skills: List[str],
    taxonomy: SkillTaxonomy,
) -> List[SkillComparisonItem]:
    """Compare a candidate's detected skills against a job's required and
    preferred skill lists.

    `candidate_skills` keys are canonical skill names the candidate has
    evidence for (from ResumeDocument.skills). Values aren't used here
    beyond membership -- evidence attachment happens in the caller.
    """
    results: List[SkillComparisonItem] = []
    candidate_names = set(candidate_skills.keys())

    def evaluate(skill: str, requirement_level: str) -> SkillComparisonItem:
        if skill in candidate_names:
            return SkillComparisonItem(
                skill=skill,
                status="MATCHED",
                requirement_level=requirement_level,
                evidence_snippets=list(getattr(candidate_skills[skill], "evidence_snippets", [])),
            )

        if CONFIG.skill_policy.allow_related_skill_partial_credit:
            related = taxonomy.related_skills(skill)
            found_related = next((r for r in related if r in candidate_names), None)
            if found_related:
                return SkillComparisonItem(
                    skill=skill,
                    status="PARTIAL",
                    requirement_level=requirement_level,
                    evidence_snippets=list(
                        getattr(candidate_skills[found_related], "evidence_snippets", [])
                    ),
                    related_skill_found=found_related,
                )

        return SkillComparisonItem(
            skill=skill,
            status="MISSING",
            requirement_level=requirement_level,
            evidence_snippets=[],
        )

    for skill in required_skills:
        results.append(evaluate(skill, "required"))
    for skill in preferred_skills:
        results.append(evaluate(skill, "preferred"))

    return results


def coverage_ratios(comparisons: List[SkillComparisonItem]) -> Dict[str, float]:
    """Compute required/preferred coverage ratios.

    A PARTIAL match counts as SkillMatchPolicy.partial_credit_weight
    (default 0.5) of a full match -- it's real signal, but must not be
    presented as equivalent to an exact match.
    """
    weight = CONFIG.skill_policy.partial_credit_weight

    def ratio(level: str) -> float:
        items = [c for c in comparisons if c.requirement_level == level]
        if not items:
            return 1.0  # no requirements of this type -> nothing missing
        score = sum(1.0 if c.status == "MATCHED" else weight if c.status == "PARTIAL" else 0.0 for c in items)
        return score / len(items)

    return {
        "required_coverage": ratio("required"),
        "preferred_coverage": ratio("preferred"),
    }
