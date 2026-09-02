"""
Recommendation engine (Section 22 / Step 15 groundwork).

Unifies two upstream sources into one prioritized, evidence-backed list:
  - skill gaps (src/recommendations/skill_gap.py)          -- job-fit issues
  - resume quality dimensions (src/nlp/resume_quality.py)  -- presentation issues

Deliberately template-based, not LLM-generated: every recommendation
string is built from a fixed template plus real extracted evidence, so
recommendations are reproducible and auditable (Engineering Rule #2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.nlp.resume_quality import ResumeQualityReport
from src.schemas import SkillGapItem

_QUALITY_SCORE_THRESHOLD = 0.6


@dataclass
class Recommendation:
    priority: str  # HIGH | MEDIUM | LOW
    category: str  # "skill_gap" | "resume_quality"
    text: str


def _skill_gap_recommendations(gaps: List[SkillGapItem]) -> List[Recommendation]:
    return [Recommendation(priority=g.priority, category="skill_gap", text=g.reason) for g in gaps]


def _quality_recommendations(quality: ResumeQualityReport) -> List[Recommendation]:
    recs: List[Recommendation] = []
    for dim in quality.dimensions:
        if dim.score >= _QUALITY_SCORE_THRESHOLD:
            continue

        if dim.name == "Quantified Impact":
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="resume_quality",
                    text=(
                        "Add measurable outcomes to more of your bullet points (e.g. '%', "
                        "'requests/day', time saved). " + dim.detail
                    ),
                )
            )
        elif dim.name == "Bullet Strength":
            evidence = "; ".join(f'"{e}"' for e in dim.evidence)
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="resume_quality",
                    text=(
                        f"Rewrite vague bullets to name what you built and its result. "
                        f"Flagged examples: {evidence}"
                        if evidence
                        else "Some bullets read as vague -- name specific technologies and outcomes."
                    ),
                )
            )
        elif dim.name == "Section Completeness":
            recs.append(
                Recommendation(
                    priority="HIGH" if dim.score < 0.5 else "LOW",
                    category="resume_quality",
                    text=f"Your resume is missing standard sections recruiters and ATS systems expect: {dim.detail}",
                )
            )
        elif dim.name == "Skill Evidence":
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="resume_quality",
                    text=(
                        "Several listed skills aren't backed by a specific example in your "
                        "experience or projects -- consider showing, not just listing, them."
                    ),
                )
            )
        elif dim.name == "Contact Info":
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="resume_quality",
                    text=f"Missing contact details may block recruiters from reaching you: {dim.detail}",
                )
            )

    return recs


def build_recommendations(gaps: List[SkillGapItem], quality: ResumeQualityReport) -> List[Recommendation]:
    all_recs = _skill_gap_recommendations(gaps) + _quality_recommendations(quality)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_recs.sort(key=lambda r: order[r.priority])
    return all_recs
