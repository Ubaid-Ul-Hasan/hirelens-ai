"""
Lightweight entity extraction for contact info and simple structured
fields (Section 8/6). Deliberately regex/rule-based -- contact info has a
highly regular surface form, so a full NER model would be overkill (see
Engineering Rule #2: don't reach for heavier tooling than the task needs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)
# very rough "City, ST" / "City, Country" heuristic
_LOCATION_RE = re.compile(r"\b([A-Z][a-zA-Z.\- ]+,\s*[A-Z]{2}\b|[A-Z][a-zA-Z.\- ]+,\s*[A-Z][a-zA-Z]+)\b")

_YEARS_EXPERIENCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s*(?:of)?\s*experience", re.IGNORECASE
)


@dataclass
class ContactInfo:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


def extract_contact_info(contact_block: str) -> ContactInfo:
    email_match = _EMAIL_RE.search(contact_block)
    phone_match = _PHONE_RE.search(contact_block)
    location_match = _LOCATION_RE.search(contact_block)

    # Name heuristic: first non-empty line that isn't itself the email/phone
    # line and doesn't look like a URL. This is intentionally conservative;
    # if it's wrong, downstream logic never depends on name for scoring.
    name = None
    for line in contact_block.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _EMAIL_RE.search(stripped) or _PHONE_RE.search(stripped) or "http" in stripped.lower():
            continue
        if len(stripped.split()) <= 5:
            name = stripped
            break

    return ContactInfo(
        name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        location=location_match.group(0) if location_match else None,
    )


def extract_years_of_experience(text: str) -> Optional[float]:
    """Look for an explicit 'X years of experience' style phrase.

    Returns None (not 0) when no such phrase is found -- absence of a
    stated number is not evidence of zero experience (Engineering Rule
    #14: "treat 'not detected' differently from 'does not exist'").
    """
    match = _YEARS_EXPERIENCE_RE.search(text)
    if match:
        return float(match.group(1))
    return None
