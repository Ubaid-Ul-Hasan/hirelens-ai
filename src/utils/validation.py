"""
Input validation for uploaded resume/job files.

Per spec Section 7, validation happens BEFORE parsing, and must clearly
distinguish between:
  - a rejected upload (wrong type / too large)
  - a successfully-read-but-empty document (e.g. a scanned PDF with no
    extractable text layer -- this is NOT the same as "no content exists"
    and must be reported to the user as needing OCR, not silently produce
    an empty analysis).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.config import CONFIG


class ValidationError(Exception):
    """Raised when an uploaded file fails hard validation (type/size)."""


@dataclass
class ExtractionQuality:
    """Result of a post-extraction content quality check.

    is_empty=True + likely_scanned=True means: the file parsed without
    error, but no usable text layer was found -- almost certainly a
    scanned/image-only PDF. The UI must surface this explicitly rather
    than proceeding as if the resume were blank.
    """

    is_empty: bool
    likely_scanned: bool
    char_count: int
    message: Optional[str] = None


def validate_file_basic(filename: str, size_bytes: int) -> None:
    """Validate extension and size before any parsing is attempted.

    Raises ValidationError with a user-facing message on failure.
    """
    ext = Path(filename).suffix.lower()
    if ext not in CONFIG.ingestion.allowed_extensions:
        allowed = ", ".join(CONFIG.ingestion.allowed_extensions)
        raise ValidationError(
            f"Unsupported file type '{ext}'. Allowed types: {allowed}."
        )

    max_bytes = CONFIG.ingestion.max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Maximum allowed is {CONFIG.ingestion.max_file_size_mb} MB."
        )

    if size_bytes == 0:
        raise ValidationError("The uploaded file is empty (0 bytes).")


def assess_extraction_quality(raw_text: str, source_ext: str) -> ExtractionQuality:
    """Check whether extracted text is usable.

    This does not raise -- it returns a structured result so the caller
    (UI / pipeline) can decide how to communicate the situation to the user.
    """
    cleaned = (raw_text or "").strip()
    char_count = len(cleaned)
    min_chars = CONFIG.ingestion.min_extractable_chars

    if char_count == 0:
        likely_scanned = source_ext == ".pdf"
        message = (
            "No extractable text was found in this PDF. It looks like a "
            "scanned/image-based document, which requires OCR (not yet "
            "implemented in this MVP). Please upload a text-based PDF, "
            "DOCX, or TXT file."
            if likely_scanned
            else "No extractable text was found in this document."
        )
        return ExtractionQuality(
            is_empty=True, likely_scanned=likely_scanned, char_count=0, message=message
        )

    if char_count < min_chars:
        return ExtractionQuality(
            is_empty=False,
            likely_scanned=False,
            char_count=char_count,
            message=(
                f"Only {char_count} characters were extracted -- this document "
                "may be incomplete, image-heavy, or unusually short. Results "
                "may be unreliable."
            ),
        )

    return ExtractionQuality(
        is_empty=False, likely_scanned=False, char_count=char_count, message=None
    )
