"""
Assembles ingestion + section_parser + entities + skill_extraction into a
single ResumeDocument (Section 6 data model).
"""
from __future__ import annotations

from typing import List, Optional

from src.nlp.entities import extract_contact_info, extract_years_of_experience
from src.nlp.section_parser import parse_sections
from src.nlp.skill_extraction import SkillTaxonomy, extract_skills
from src.schemas import EducationEntry, ExperienceEntry, ProjectEntry, ResumeDocument, SkillEvidence
from src.utils.logging import get_logger

logger = get_logger("nlp.resume_parser")


def _split_into_entries(section_text: str) -> List[str]:
    """Split a section's raw text into rough per-entry blocks.

    Resumes vary wildly in format, so this uses a conservative heuristic:
    a blank line starts a new entry. This is intentionally simple -- it is
    NOT meant to perfectly parse every resume layout, and downstream logic
    treats these as "raw_text" blobs (Section 6's ExperienceEntry.raw_text)
    rather than depending on perfect field-level splitting.
    """
    blocks = [b.strip() for b in section_text.split("\n\n") if b.strip()]
    if len(blocks) <= 1:
        # no blank-line separation found; fall back to treating the whole
        # section as one entry rather than guessing incorrectly
        return [section_text.strip()] if section_text.strip() else []
    return blocks


def _entry_to_experience(block: str) -> ExperienceEntry:
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    title = lines[0] if lines else None
    bullets = [l.lstrip("-*\u2022 ").strip() for l in lines[1:]]
    is_current = "present" in block.lower() or "current" in block.lower()
    return ExperienceEntry(title=title, bullets=bullets, is_current=is_current, raw_text=block)


def _entry_to_education(block: str) -> EducationEntry:
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    degree = lines[0] if lines else None
    return EducationEntry(degree=degree, raw_text=block)


def _entry_to_project(block: str) -> ProjectEntry:
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    name = lines[0] if lines else None
    bullets = [l.lstrip("-*\u2022 ").strip() for l in lines[1:]]
    return ProjectEntry(name=name, bullets=bullets, raw_text=block)


def parse_resume(
    raw_text: str,
    taxonomy: SkillTaxonomy,
    source_filename: Optional[str] = None,
    extraction_warning: Optional[str] = None,
) -> ResumeDocument:
    parsed_sections = parse_sections(raw_text)
    contact = extract_contact_info(parsed_sections.contact_block)

    experience_blocks = _split_into_entries(parsed_sections.sections.get("experience", ""))
    education_blocks = _split_into_entries(parsed_sections.sections.get("education", ""))
    project_blocks = _split_into_entries(parsed_sections.sections.get("projects", ""))

    experience = [_entry_to_experience(b) for b in experience_blocks]
    education = [_entry_to_education(b) for b in education_blocks]
    projects = [_entry_to_project(b) for b in project_blocks]

    certifications_text = parsed_sections.sections.get("certifications", "")
    certifications = [l.strip("-*\u2022 ").strip() for l in certifications_text.split("\n") if l.strip()]

    achievements_text = parsed_sections.sections.get("achievements", "")
    achievements = [l.strip("-*\u2022 ").strip() for l in achievements_text.split("\n") if l.strip()]

    # Skill extraction runs per-section so we know WHERE a skill was found
    # (Section 6: SkillEvidence.source_section), then results are merged,
    # keeping the highest confidence and combining evidence when the same
    # skill appears in multiple sections.
    merged_skills: dict[str, SkillEvidence] = {}
    for section_name, section_text in parsed_sections.sections.items():
        if not section_text.strip():
            continue
        detections = extract_skills(section_text, taxonomy, source_section=section_name)
        for d in detections:
            if d.skill not in merged_skills:
                merged_skills[d.skill] = SkillEvidence(
                    skill=d.skill,
                    raw_mentions=list(d.raw_mentions),
                    evidence_snippets=list(d.evidence_snippets),
                    source_section=d.source_section,
                    confidence=d.confidence,
                )
            else:
                existing = merged_skills[d.skill]
                existing.confidence = max(existing.confidence, d.confidence)
                for m in d.raw_mentions:
                    if m not in existing.raw_mentions:
                        existing.raw_mentions.append(m)
                for e in d.evidence_snippets:
                    if e not in existing.evidence_snippets:
                        existing.evidence_snippets.append(e)

    total_years = extract_years_of_experience(raw_text)

    return ResumeDocument(
        raw_text=raw_text,
        name=contact.name,
        email=contact.email,
        phone=contact.phone,
        location=contact.location,
        summary=parsed_sections.sections.get("summary"),
        skills=list(merged_skills.values()),
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        sections=parsed_sections.sections,
        total_years_experience=total_years,
        source_filename=source_filename,
        extraction_warning=extraction_warning,
    )
