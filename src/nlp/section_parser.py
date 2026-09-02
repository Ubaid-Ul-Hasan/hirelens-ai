"""
Resume section detection (Section 8).

Hybrid approach as specified:
  1. heading detection (line looks like a heading: short, title-case/caps,
     no sentence punctuation, on its own line)
  2. lexical rules (heading text matches a known synonym list)
  3. layout/order heuristics (contact info is assumed to precede the first
     detected heading)
  4. NLP assistance is intentionally NOT used here in the MVP -- rules
     handle the common case, keeping this stage fast, deterministic, and
     easy to unit test. (Optional spaCy-assisted heading disambiguation is
     a documented extension point, not required for correctness.)

We do NOT assume a fixed resume layout; any heading synonym list can map to
the same canonical section, and unmatched headings just become part of the
'other' bucket rather than crashing the parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CANONICAL_SECTIONS = [
    "contact",
    "summary",
    "skills",
    "education",
    "experience",
    "projects",
    "certifications",
    "achievements",
    "publications",
    "awards",
]

# Synonym lists per canonical section (Section 8 example: "Work Experience",
# "Professional Experience", "Employment", "Experience" -> "experience").
_SECTION_SYNONYMS: Dict[str, List[str]] = {
    "summary": ["summary", "professional summary", "profile", "objective", "about me", "career objective"],
    "skills": ["skills", "technical skills", "core competencies", "skill set", "technologies", "tech stack"],
    "education": ["education", "academic background", "education & training", "academic qualifications"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "relevant experience",
    ],
    "projects": ["projects", "personal projects", "key projects", "project experience", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses", "licenses & certifications"],
    "achievements": ["achievements", "accomplishments", "honors and achievements"],
    "publications": ["publications", "papers", "research publications"],
    "awards": ["awards", "honors", "awards & honors"],
}

# Build a reverse lookup: normalized synonym text -> canonical section
_SYNONYM_TO_CANONICAL: Dict[str, str] = {
    syn: canonical for canonical, syns in _SECTION_SYNONYMS.items() for syn in syns
}

_MAX_HEADING_WORDS = 5


@dataclass
class ParsedSections:
    sections: Dict[str, str] = field(default_factory=dict)  # canonical -> text
    contact_block: str = ""  # text before the first detected heading
    unrecognized_headings: List[str] = field(default_factory=list)


def _looks_like_heading(line: str) -> bool:
    """Heuristic: a heading is short, has no terminal sentence punctuation,
    and is either ALL CAPS, Title Case, or ends without a period -- the
    kind of line that stands alone as a label rather than prose.
    """
    stripped = line.strip().strip(":").strip()
    if not stripped:
        return False
    word_count = len(stripped.split())
    if word_count == 0 or word_count > _MAX_HEADING_WORDS:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    # reject lines that look like bullet points or contain typical bullet
    # markers / long punctuation runs common in prose, not headings
    if re.match(r"^[\u2022\-\*]\s", line.strip()):
        return False
    return True


def _normalize_heading(line: str) -> str:
    return line.strip().strip(":").strip().lower()


def parse_sections(raw_text: str) -> ParsedSections:
    """Split resume raw text into canonical sections.

    Any text before the first recognized heading is treated as the
    contact/header block (name, email, phone, location -- Section 8's
    'layout/order heuristic': contact info precedes the first section).
    """
    lines = raw_text.split("\n")

    result = ParsedSections()
    current_canonical: Optional[str] = None
    current_buffer: List[str] = []
    contact_buffer: List[str] = []
    seen_first_heading = False

    def flush():
        if current_canonical and current_buffer:
            existing = result.sections.get(current_canonical, "")
            joined = "\n".join(current_buffer).strip()
            result.sections[current_canonical] = (existing + "\n" + joined).strip() if existing else joined

    for line in lines:
        if _looks_like_heading(line):
            normalized = _normalize_heading(line)
            canonical = _SYNONYM_TO_CANONICAL.get(normalized)
            if canonical:
                flush()
                current_canonical = canonical
                current_buffer = []
                seen_first_heading = True
                continue
            else:
                # Not a recognized section synonym. If we're not inside a
                # section yet, it's just part of the contact block (e.g. a
                # short name line); if we ARE inside a section, treat it as
                # unrecognized-but-log it, and keep it as body text of the
                # current section rather than dropping it.
                if not seen_first_heading:
                    contact_buffer.append(line)
                    continue
                elif _looks_like_heading(line) and len(line.strip().split()) <= 3:
                    result.unrecognized_headings.append(line.strip())

        if seen_first_heading:
            current_buffer.append(line)
        else:
            contact_buffer.append(line)

    flush()
    result.contact_block = "\n".join(contact_buffer).strip()
    return result
