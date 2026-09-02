"""
Job description parsing (Section 9).

Key requirement from the spec: don't confuse every noun in a JD with a
required skill, and separate "required" from "preferred" (nice-to-have,
bonus, etc.) rather than lumping all mentioned skills together.

Approach:
  1. Split the JD into rough sections using heading detection (a JD
     usually has "Requirements" / "Responsibilities" / "Qualifications" /
     "Nice to Have" blocks; less structured than resumes, so we fall back
     to scanning the whole text for skills if no headings are found).
  2. Within a "preferred/nice-to-have/bonus" flavored section, all skills
     found are preferred. Within "requirements/qualifications" or the
     whole document (when unstructured), skills are required BY DEFAULT
     unless found in a bonus subsection, or unless a line explicitly says
     "preferred"/"nice to have"/"bonus" inline.
  3. Responsibilities are extracted from a "responsibilities/duties/what
     you'll do" section, kept as bullet-level strings rather than trying
     to interpret them into structured skills.
"""
from __future__ import annotations

import re
from typing import List, Optional

from src.nlp.skill_extraction import SkillTaxonomy, extract_skills
from src.schemas import JobDocument
from src.utils.logging import get_logger

logger = get_logger("nlp.job_parser")

_REQUIRED_HEADING_RE = re.compile(
    r"^(requirements?|qualifications?|required skills?|must[- ]have|minimum qualifications?)\s*:?\s*$",
    re.IGNORECASE,
)
_PREFERRED_HEADING_RE = re.compile(
    r"^(preferred|nice[- ]to[- ]have|bonus( points)?|desired skills?|preferred qualifications?)\s*:?\s*$",
    re.IGNORECASE,
)
_RESPONSIBILITIES_HEADING_RE = re.compile(
    r"^(responsibilities|duties|what you.?ll do|the role|key responsibilities)\s*:?\s*$",
    re.IGNORECASE,
)
_EDUCATION_HEADING_RE = re.compile(r"^(education|academic requirements?)\s*:?\s*$", re.IGNORECASE)

_INLINE_PREFERRED_RE = re.compile(r"\b(preferred|nice to have|bonus|a plus|desired)\b", re.IGNORECASE)

_YEARS_RANGE_RE = re.compile(r"(\d+)\s*(?:-|to|\+)?\s*(\d+)?\+?\s*years?", re.IGNORECASE)
_TITLE_RE = re.compile(r"^([A-Z][A-Za-z0-9/&+\- ]{2,60})$")


def _classify_lines(raw_text: str):
    """Walk lines and bucket them by which heading section they fall
    under. Returns dict of section_name -> list[str] lines.
    """
    buckets = {"required": [], "preferred": [], "responsibilities": [], "education": [], "other": []}
    current = "other"
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _REQUIRED_HEADING_RE.match(stripped):
            current = "required"
            continue
        if _PREFERRED_HEADING_RE.match(stripped):
            current = "preferred"
            continue
        if _RESPONSIBILITIES_HEADING_RE.match(stripped):
            current = "responsibilities"
            continue
        if _EDUCATION_HEADING_RE.match(stripped):
            current = "education"
            continue
        buckets[current].append(stripped)
    return buckets


_ANY_KNOWN_HEADING_RE_LIST = [
    _REQUIRED_HEADING_RE,
    _PREFERRED_HEADING_RE,
    _RESPONSIBILITIES_HEADING_RE,
    _EDUCATION_HEADING_RE,
]


def _guess_title(raw_text: str) -> Optional[str]:
    """First non-empty line, if it looks like a plausible job title (short,
    no terminal punctuation, and NOT itself a recognized section heading
    like "Requirements") -- best-effort only, never blocks parsing.
    """
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _ANY_KNOWN_HEADING_RE_LIST):
            return None  # JD starts directly with a heading -- no title to guess
        if len(stripped.split()) <= 8 and not stripped.endswith((".", ":")):
            return stripped
        return None
    return None


def _extract_min_years(text: str) -> Optional[float]:
    match = _YEARS_RANGE_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def parse_job_description(
    raw_text: str,
    taxonomy: SkillTaxonomy,
    source_filename: Optional[str] = None,
) -> JobDocument:
    buckets = _classify_lines(raw_text)

    # If no explicit "required" heading was ever found, treat "other" as
    # the effective requirements pool (most JDs just list skills under a
    # generic "Requirements" or even no heading at all).
    has_structured_required = len(buckets["required"]) > 0
    required_pool_lines = buckets["required"] if has_structured_required else buckets["other"]

    required_text = "\n".join(required_pool_lines)
    preferred_text = "\n".join(buckets["preferred"])

    # skills explicitly under a "preferred" heading
    preferred_detections = extract_skills(preferred_text, taxonomy, source_section="preferred")
    preferred_skill_names = {d.skill for d in preferred_detections}

    # skills in the required pool, MINUS any line that inline-flags itself
    # as preferred/nice-to-have/bonus even though it wasn't under a
    # dedicated heading
    required_lines_excluding_inline_preferred = [
        line for line in required_pool_lines if not _INLINE_PREFERRED_RE.search(line)
    ]
    inline_preferred_lines = [line for line in required_pool_lines if _INLINE_PREFERRED_RE.search(line)]

    required_detections = extract_skills(
        "\n".join(required_lines_excluding_inline_preferred), taxonomy, source_section="required"
    )
    inline_preferred_detections = extract_skills(
        "\n".join(inline_preferred_lines), taxonomy, source_section="preferred"
    )

    required_skill_names = {d.skill for d in required_detections}
    preferred_skill_names |= {d.skill for d in inline_preferred_detections}

    # a skill should not appear as both required and preferred; required
    # wins only if it was found in the dedicated required pool AND never
    # flagged preferred anywhere -- otherwise treat as preferred (more
    # conservative: don't overstate what's mandatory)
    required_skill_names -= preferred_skill_names

    responsibilities = [l for l in buckets["responsibilities"] if len(l.split()) > 2]
    education_requirements = [l for l in buckets["education"] if len(l.split()) > 1]

    min_years = _extract_min_years(raw_text)
    experience_requirements = []
    if min_years is not None:
        experience_requirements.append(f"{min_years:.0f}+ years of experience (detected)")

    title = _guess_title(raw_text)

    return JobDocument(
        raw_text=raw_text,
        title=title,
        required_skills=sorted(required_skill_names),
        preferred_skills=sorted(preferred_skill_names),
        responsibilities=responsibilities,
        experience_requirements=experience_requirements,
        min_years_experience=min_years,
        education_requirements=education_requirements,
        sections={k: "\n".join(v) for k, v in buckets.items() if v},
        source_filename=source_filename,
    )
