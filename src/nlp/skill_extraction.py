"""
Skill extraction engine (Section 10).

Rule/dictionary-based on purpose: the spec explicitly asks us not to reach
for an LLM when a deterministic approach solves the task (Engineering Rule
#2), and skill detection from a controlled taxonomy is exactly that case.
This also keeps the system fully explainable -- every detected skill can
point back at the exact sentence that triggered it.

Design choices:
  - Matching is done on word boundaries over a normalized (lowercased,
    punctuation-tolerant) version of the text, so "sklearn" matches but
    "asklearner" does not.
  - A skill is only ever added once per document, but every raw surface
    form + surrounding sentence is retained as evidence.
  - We deliberately do NOT upgrade a passing mention into a strong claim.
    Confidence is lower for skills only mentioned once, in passing,
    outside a "Skills" section (Section 10: "should not automatically
    become 'AWS expert'").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import TAXONOMY_PATH
from src.utils.logging import get_logger

logger = get_logger("nlp.skill_extraction")

# Words that suggest a mention is exploratory/tangential rather than a
# confident skill claim -- used only to adjust confidence, never to
# suppress detection entirely (we still want gaps analysis to know the
# word appeared at all).
_HEDGE_PATTERNS = [
    r"\bexposure to\b",
    r"\bfamiliar(?:ity)? with\b",
    r"\bworked with\b",
    r"\bsome experience with\b",
    r"\bbasic\b",
]


@dataclass
class SkillDefinition:
    canonical: str
    category: str
    aliases: List[str]
    related: List[str]
    # compiled regexes for each alias (word-boundary, case-insensitive)
    patterns: List[re.Pattern] = field(default_factory=list, repr=False)


@dataclass
class DetectedSkill:
    skill: str
    raw_mentions: List[str]
    evidence_snippets: List[str]
    confidence: float
    source_section: Optional[str] = None


class SkillTaxonomy:
    """Loads and indexes the skill taxonomy JSON for fast lookup."""

    def __init__(self, taxonomy_path: Path = TAXONOMY_PATH):
        self.taxonomy_path = taxonomy_path
        self.skills: Dict[str, SkillDefinition] = {}
        self._load()

    def _load(self) -> None:
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for category, skills in raw.get("categories", {}).items():
            for canonical, info in skills.items():
                aliases = info.get("aliases", [])
                # canonical name itself is always a valid surface form
                all_surface_forms = list({canonical.lower(), *[a.lower() for a in aliases]})
                patterns = [
                    re.compile(rf"(?<![a-zA-Z0-9]){re.escape(form)}(?![a-zA-Z0-9])", re.IGNORECASE)
                    for form in all_surface_forms
                ]
                self.skills[canonical] = SkillDefinition(
                    canonical=canonical,
                    category=category,
                    aliases=aliases,
                    related=info.get("related", []),
                    patterns=patterns,
                )
        logger.info("Loaded %d skills across %d categories", len(self.skills), len(raw.get("categories", {})))

    def all_canonical_names(self) -> List[str]:
        return list(self.skills.keys())

    def related_skills(self, canonical: str) -> List[str]:
        skill = self.skills.get(canonical)
        return skill.related if skill else []

    def normalize(self, name: str) -> Optional[str]:
        """Map a free-text skill name (e.g. from a job description's
        'Required Skills' bullet list) to its canonical taxonomy name, if
        recognized. Returns None if not found in the taxonomy.
        """
        name_lower = name.strip().lower()
        for canonical, definition in self.skills.items():
            if name_lower == canonical.lower() or name_lower in [a.lower() for a in definition.aliases]:
                return canonical
        return None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_skills(
    text: str,
    taxonomy: SkillTaxonomy,
    source_section: Optional[str] = None,
    max_evidence_per_skill: int = 3,
) -> List[DetectedSkill]:
    """Scan `text` for every skill in the taxonomy and return detections
    with evidence. `text` should be a single section's text (e.g. just the
    "Experience" section) so `source_section` is accurate; call this once
    per section and merge results at a higher level for whole-document
    extraction.
    """
    sentences = _split_sentences(text)
    detections: Dict[str, DetectedSkill] = {}

    for sentence in sentences:
        for canonical, definition in taxonomy.skills.items():
            for pattern in definition.patterns:
                match = pattern.search(sentence)
                if not match:
                    continue

                confidence = 1.0
                for hedge in _HEDGE_PATTERNS:
                    if re.search(hedge, sentence, re.IGNORECASE):
                        confidence = 0.6
                        break

                if canonical not in detections:
                    detections[canonical] = DetectedSkill(
                        skill=canonical,
                        raw_mentions=[],
                        evidence_snippets=[],
                        confidence=confidence,
                        source_section=source_section,
                    )

                detected = detections[canonical]
                if match.group(0) not in detected.raw_mentions:
                    detected.raw_mentions.append(match.group(0))
                if (
                    sentence not in detected.evidence_snippets
                    and len(detected.evidence_snippets) < max_evidence_per_skill
                ):
                    detected.evidence_snippets.append(sentence)
                # once a skill is seen with high confidence anywhere, keep
                # the highest confidence observed rather than the last one
                detected.confidence = max(detected.confidence, confidence)
                break  # don't double count multiple aliases of same skill in one sentence

    return list(detections.values())
