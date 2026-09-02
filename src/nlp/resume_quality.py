"""
Resume quality analyzer (Section 20-ish / Step 13).

Independent of any specific job description -- this measures whether the
resume itself is well-constructed: does it have the sections an ATS/
recruiter expects, does it show quantified impact, is there evidence
behind claimed skills, etc. This is deliberately separate from
job-match scoring (src/scoring/scoring_engine.py) because a resume can be
well-written but a poor fit for a specific role, or a strong fit but
poorly presented -- conflating the two would hide which problem the
candidate actually has.

All checks are rule-based and each one records exactly what triggered it,
so "your bullets lack quantified impact" always comes with a list of the
actual weak bullets, never a bare score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.nlp.weak_bullets import find_weak_bullets
from src.schemas import ResumeDocument

_EXPECTED_SECTIONS = ["summary", "skills", "experience", "education"]
_BONUS_SECTIONS = ["projects", "certifications", "achievements"]

_QUANTIFICATION_RE = re.compile(
    r"(\d+(\.\d+)?\s*%|\$\s?\d|\b\d+(\.\d+)?\s*(x|k|m|ms|s|sec|seconds|users?|requests?|records?|"
    r"customers?|million|billion|thousand)\b)",
    re.IGNORECASE,
)


@dataclass
class QualityDimension:
    name: str
    score: float  # 0-1
    detail: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ResumeQualityReport:
    overall_score: float
    dimensions: List[QualityDimension]


def _section_completeness(resume: ResumeDocument) -> QualityDimension:
    present = [s for s in _EXPECTED_SECTIONS if resume.sections.get(s, "").strip()]
    bonus_present = [s for s in _BONUS_SECTIONS if resume.sections.get(s, "").strip()]
    missing = [s for s in _EXPECTED_SECTIONS if s not in present]

    score = (len(present) / len(_EXPECTED_SECTIONS)) * 0.85 + (len(bonus_present) / len(_BONUS_SECTIONS)) * 0.15
    detail = (
        f"{len(present)}/{len(_EXPECTED_SECTIONS)} core sections present"
        + (f"; missing: {', '.join(missing)}" if missing else "")
    )
    return QualityDimension(name="Section Completeness", score=round(min(1.0, score), 3), detail=detail, evidence=missing)


def _quantified_impact(resume: ResumeDocument) -> QualityDimension:
    all_bullets = [b for e in resume.experience for b in e.bullets] + [b for p in resume.projects for b in p.bullets]
    if not all_bullets:
        return QualityDimension(
            name="Quantified Impact", score=0.4, detail="No experience/project bullets detected to evaluate."
        )

    quantified = [b for b in all_bullets if _QUANTIFICATION_RE.search(b)]
    ratio = len(quantified) / len(all_bullets)
    return QualityDimension(
        name="Quantified Impact",
        score=round(ratio, 3),
        detail=f"{len(quantified)}/{len(all_bullets)} bullets include a measurable number or metric.",
        evidence=quantified[:3],
    )


def _skill_evidence_depth(resume: ResumeDocument) -> QualityDimension:
    if not resume.skills:
        return QualityDimension(name="Skill Evidence", score=0.3, detail="No skills detected in the resume at all.")

    with_context_evidence = [
        s for s in resume.skills if s.source_section in ("experience", "projects") or len(s.evidence_snippets) > 1
    ]
    ratio = len(with_context_evidence) / len(resume.skills)
    return QualityDimension(
        name="Skill Evidence",
        score=round(ratio, 3),
        detail=(
            f"{len(with_context_evidence)}/{len(resume.skills)} listed skills are backed by experience/"
            "project evidence, not just a bare skills list."
        ),
    )


def _weak_bullet_dimension(resume: ResumeDocument) -> QualityDimension:
    all_bullets = [b for e in resume.experience for b in e.bullets] + [b for p in resume.projects for b in p.bullets]
    weak = find_weak_bullets(all_bullets)
    if not all_bullets:
        return QualityDimension(name="Bullet Strength", score=0.4, detail="No bullets to evaluate.")
    ratio_strong = 1.0 - (len(weak) / len(all_bullets))
    return QualityDimension(
        name="Bullet Strength",
        score=round(max(0.0, ratio_strong), 3),
        detail=f"{len(weak)}/{len(all_bullets)} bullets read as vague (weak verb, no outcome or technology named).",
        evidence=[w.text for w in weak[:3]],
    )


def _contact_completeness(resume: ResumeDocument) -> QualityDimension:
    fields_present = sum(1 for v in [resume.name, resume.email, resume.phone] if v)
    return QualityDimension(
        name="Contact Info",
        score=round(fields_present / 3, 3),
        detail=f"{fields_present}/3 of name/email/phone were detected.",
    )


def analyze_resume_quality(resume: ResumeDocument) -> ResumeQualityReport:
    dimensions = [
        _section_completeness(resume),
        _quantified_impact(resume),
        _skill_evidence_depth(resume),
        _weak_bullet_dimension(resume),
        _contact_completeness(resume),
    ]
    overall = sum(d.score for d in dimensions) / len(dimensions)
    return ResumeQualityReport(overall_score=round(overall, 3), dimensions=dimensions)
