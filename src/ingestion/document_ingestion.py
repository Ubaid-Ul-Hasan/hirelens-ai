"""
Single entry point for turning an uploaded file into raw text.

Implements the pipeline from spec Section 7:
    File -> extension validation -> size/content validation ->
    parser selection -> raw text extraction -> whitespace normalization ->
    quality checks
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ingestion.docx_parser import parse_docx
from src.ingestion.txt_parser import parse_txt
from src.utils.logging import get_logger
from src.utils.validation import (
    ExtractionQuality,
    ValidationError,
    assess_extraction_quality,
    validate_file_basic,
)

logger = get_logger("ingestion")


@dataclass
class IngestionResult:
    raw_text: str
    source_filename: str
    extension: str
    quality: ExtractionQuality


def _normalize_whitespace(text: str) -> str:
    """Conservative whitespace normalization.

    Collapses runs of 3+ blank lines to a single blank line and strips
    trailing whitespace on each line, but does NOT collapse all whitespace
    (that would destroy paragraph/section boundaries the section parser
    relies on).
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def ingest_document(
    file_bytes: bytes, filename: str, size_bytes: Optional[int] = None
) -> IngestionResult:
    """Validate, parse, and quality-check an uploaded resume/JD file.

    Raises ValidationError for hard failures (bad extension, oversized,
    zero-byte file, corrupt document). Soft issues (empty/likely-scanned
    text, unusually short content) are returned in `.quality` for the
    caller to display rather than raised as exceptions.
    """
    size_bytes = size_bytes if size_bytes is not None else len(file_bytes)
    validate_file_basic(filename, size_bytes)

    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        raw_text = parse_txt(file_bytes)
    elif ext == ".docx":
        raw_text = parse_docx(file_bytes)
    elif ext == ".pdf":
        # imported lazily so environments without PyMuPDF can still use
        # the txt/docx paths (e.g. this sandbox, or a minimal deployment).
        from src.ingestion.pdf_parser import parse_pdf

        raw_text = parse_pdf(file_bytes)
    else:
        # validate_file_basic should have already caught this, but this
        # guards against future extensions being added to the allow-list
        # without a matching parser branch.
        raise ValidationError(f"No parser implemented for extension '{ext}'.")

    raw_text = _normalize_whitespace(raw_text)
    quality = assess_extraction_quality(raw_text, ext)

    if quality.message:
        logger.info("Extraction quality note for %s: %s", filename, quality.message)

    return IngestionResult(
        raw_text=raw_text,
        source_filename=filename,
        extension=ext,
        quality=quality,
    )
