"""
Weak bullet-point detection (Step 14).

A bullet is flagged "weak" when it opens with a vague responsibility verb
("responsible for", "helped with", "worked on") AND contains neither a
quantified outcome (a number/percent/metric) NOR a named technology/skill
-- i.e. it describes an activity without evidence of impact or specifics.
A bullet with either signal present is left alone, even if it also uses a
vague opener, since one strong signal is enough to be useful to a reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_VAGUE_OPENERS = [
    r"responsible for",
    r"helped (with|to)?",
    r"worked on",
    r"involved in",
    r"assisted (with|in)?",
    r"participated in",
    r"tasked with",
]
_VAGUE_OPENER_RE = re.compile(r"^(" + "|".join(_VAGUE_OPENERS) + r")\b", re.IGNORECASE)

_QUANTIFICATION_RE = re.compile(
    r"(\d+(\.\d+)?\s*%|\$\s?\d|\b\d+(\.\d+)?\s*(x|k|m|ms|s|sec|seconds|users?|requests?|records?|"
    r"customers?|million|billion|thousand)\b)",
    re.IGNORECASE,
)

# A very small set of common technical nouns; this deliberately reuses the
# resume's own detected skill vocabulary at call time rather than a fixed
# list wherever possible (see find_weak_bullets' `known_skill_terms` arg).


@dataclass
class WeakBulletFlag:
    text: str
    reason: str


def _has_named_technology(bullet: str, known_skill_terms: List[str]) -> bool:
    lowered = bullet.lower()
    return any(term.lower() in lowered for term in known_skill_terms)


def find_weak_bullets(bullets: List[str], known_skill_terms: List[str] = None) -> List[WeakBulletFlag]:
    known_skill_terms = known_skill_terms or []
    flags: List[WeakBulletFlag] = []

    for bullet in bullets:
        stripped = bullet.strip()
        if not stripped:
            continue

        has_vague_opener = bool(_VAGUE_OPENER_RE.match(stripped))
        has_metric = bool(_QUANTIFICATION_RE.search(stripped))
        has_tech = _has_named_technology(stripped, known_skill_terms)
        is_very_short = len(stripped.split()) <= 4

        if (has_vague_opener or is_very_short) and not has_metric and not has_tech:
            reason = (
                "Vague opener with no measurable outcome or named technology"
                if has_vague_opener
                else "Very short bullet with no measurable outcome or named technology"
            )
            flags.append(WeakBulletFlag(text=stripped, reason=reason))

    return flags
