"""
Section-level semantic matching (Section 13/14, Step 9).

Compares specific resume sections (summary, experience, projects) against
the job description using embeddings, rather than only comparing whole
documents -- this is what lets the explanation say "your Experience
section aligns well with the role's responsibilities" instead of a single
opaque number.

Every result explicitly reports whether it was computed from real
embeddings or is unavailable -- callers must never silently substitute a
different number without saying so (Engineering Rule #19).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.embeddings.encoder import cosine_sim, encode, is_available
from src.schemas import JobDocument, ResumeDocument
from src.utils.logging import get_logger

logger = get_logger("matching.semantic")

# Which resume sections to compare against which job text, in priority
# order for the explanation UI.
_RESUME_SECTIONS_TO_COMPARE = ["summary", "experience", "projects", "skills"]


@dataclass
class SectionSemanticResult:
    section: str
    similarity: Optional[float]  # None if embeddings unavailable
    available: bool


@dataclass
class SemanticMatchResult:
    overall_similarity: Optional[float]
    section_results: List[SectionSemanticResult]
    available: bool
    unavailable_reason: Optional[str] = None


def compute_semantic_match(resume: ResumeDocument, job: JobDocument) -> SemanticMatchResult:
    if not is_available():
        from src.embeddings.encoder import unavailability_reason

        return SemanticMatchResult(
            overall_similarity=None,
            section_results=[
                SectionSemanticResult(section=s, similarity=None, available=False)
                for s in _RESUME_SECTIONS_TO_COMPARE
            ],
            available=False,
            unavailable_reason=unavailability_reason() or "embedding model not loaded",
        )

    job_text = job.raw_text
    texts_to_encode: List[str] = [resume.raw_text, job_text]
    section_texts: Dict[str, str] = {}
    for section_name in _RESUME_SECTIONS_TO_COMPARE:
        text = resume.sections.get(section_name, "")
        if text.strip():
            section_texts[section_name] = text
            texts_to_encode.append(text)

    vectors = encode(texts_to_encode)
    if vectors is None:
        # model became unavailable between the is_available() check and now
        # (e.g. transient failure) -- degrade the same way as the upfront check
        return SemanticMatchResult(
            overall_similarity=None,
            section_results=[
                SectionSemanticResult(section=s, similarity=None, available=False)
                for s in _RESUME_SECTIONS_TO_COMPARE
            ],
            available=False,
            unavailable_reason="embedding encode() returned no vectors",
        )

    resume_vec, job_vec = vectors[0], vectors[1]
    overall = cosine_sim(resume_vec, job_vec)

    section_results: List[SectionSemanticResult] = []
    offset = 2
    for section_name in _RESUME_SECTIONS_TO_COMPARE:
        if section_name in section_texts:
            sec_vec = vectors[offset]
            offset += 1
            sim = cosine_sim(sec_vec, job_vec)
            section_results.append(SectionSemanticResult(section=section_name, similarity=sim, available=True))
        else:
            section_results.append(SectionSemanticResult(section=section_name, similarity=None, available=False))

    return SemanticMatchResult(overall_similarity=overall, section_results=section_results, available=True)
