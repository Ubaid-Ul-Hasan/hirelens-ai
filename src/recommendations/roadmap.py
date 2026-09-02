"""
Personalized skill roadmap generator (Step 15).

Turns the prioritized skill-gap list into a concrete, time-boxed learning
plan. Suggested resources are DELIBERATELY generic/type-level ("official
documentation", "a hands-on project") rather than named courses or URLs --
naming specific external courses would go stale immediately and isn't
something this system can verify is still accurate, current, or even
still exists. This keeps every claim in the roadmap something the system
can actually stand behind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.schemas import SkillGapItem

# Suggested learning approach by rough skill category. Falls back to a
# generic template for any skill not covered here.
_CATEGORY_HINTS: Dict[str, str] = {
    "Docker": "containerize a small existing project end-to-end (Dockerfile + docker-compose)",
    "Kubernetes": "deploy a containerized app to a local cluster (e.g. minikube) and configure a service + deployment",
    "AWS": "complete a hands-on project using at least 2 core services (e.g. S3 + Lambda, or EC2 + RDS)",
    "Azure": "complete a hands-on project using core compute + storage services",
    "GCP": "complete a hands-on project using core compute + storage services",
    "PyTorch": "reimplement a small model architecture from scratch and train it on a public dataset",
    "TensorFlow": "reimplement a small model architecture from scratch and train it on a public dataset",
    "FastAPI": "wrap an existing script or model as a REST API with at least 2 endpoints and tests",
    "SQL": "practice writing joins/aggregations/window functions against a real public dataset",
    "MLflow": "add experiment tracking to an existing personal ML project",
    "Spark": "process a multi-GB public dataset locally with PySpark and profile the job",
}

_DEFAULT_HINT = "build or contribute to a small project that requires using this skill directly"


@dataclass
class RoadmapItem:
    week_label: str
    skill: str
    priority: str
    action: str


@dataclass
class SkillRoadmap:
    items: List[RoadmapItem] = field(default_factory=list)
    note: str = (
        "This roadmap suggests what to practice and in what order, based on the skills "
        "the target role asks for that weren't detected in your resume. It intentionally "
        "does not recommend specific named courses or paid resources."
    )


def build_roadmap(gaps: List[SkillGapItem], weeks_per_skill: int = 2) -> SkillRoadmap:
    """Builds a simple sequential roadmap: HIGH-priority gaps first, then
    MEDIUM, then LOW, `weeks_per_skill` weeks allocated to each.
    """
    ordered = sorted(gaps, key=lambda g: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[g.priority])

    items: List[RoadmapItem] = []
    week_cursor = 1
    for gap in ordered:
        hint = _CATEGORY_HINTS.get(gap.skill, _DEFAULT_HINT)
        week_label = (
            f"Week {week_cursor}"
            if weeks_per_skill == 1
            else f"Weeks {week_cursor}-{week_cursor + weeks_per_skill - 1}"
        )
        action = f"Learn the fundamentals of {gap.skill}, then {hint}."
        items.append(RoadmapItem(week_label=week_label, skill=gap.skill, priority=gap.priority, action=action))
        week_cursor += weeks_per_skill

    return SkillRoadmap(items=items)
