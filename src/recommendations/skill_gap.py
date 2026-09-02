"""
Skill gap engine (Section 18).

Turns the skill comparison list into a prioritized gap list:
  HIGH   -> missing required skill
  MEDIUM -> weak/partial required skill (a related skill was found, but
            not the exact one asked for)
  LOW    -> missing preferred skill

Partial preferred-skill matches are not reported as gaps at all -- the
candidate has meaningful related signal for a nice-to-have, which isn't
worth flagging as something to fix.
"""
from __future__ import annotations

from typing import List

from src.schemas import SkillComparisonItem, SkillGapItem


def build_skill_gaps(comparisons: List[SkillComparisonItem]) -> List[SkillGapItem]:
    gaps: List[SkillGapItem] = []

    for item in comparisons:
        if item.status == "MATCHED":
            continue

        if item.requirement_level == "required" and item.status == "MISSING":
            gaps.append(
                SkillGapItem(
                    skill=item.skill,
                    requirement_level="required",
                    priority="HIGH",
                    reason=f"{item.skill} is a required skill for this role but was not detected in the resume.",
                )
            )
        elif item.requirement_level == "required" and item.status == "PARTIAL":
            gaps.append(
                SkillGapItem(
                    skill=item.skill,
                    requirement_level="required",
                    priority="MEDIUM",
                    reason=(
                        f"{item.skill} is required, but only a related skill "
                        f"({item.related_skill_found}) was detected -- consider making "
                        f"{item.skill} experience explicit if you genuinely have it."
                    ),
                )
            )
        elif item.requirement_level == "preferred" and item.status == "MISSING":
            gaps.append(
                SkillGapItem(
                    skill=item.skill,
                    requirement_level="preferred",
                    priority="LOW",
                    reason=f"{item.skill} is listed as a preferred/nice-to-have skill and was not detected.",
                )
            )
        # PARTIAL + preferred: intentionally not flagged as a gap (see docstring)

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    gaps.sort(key=lambda g: priority_order[g.priority])
    return gaps
