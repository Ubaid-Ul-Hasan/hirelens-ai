"""PDF ingestion via PyMuPDF (fitz).

NOTE: not executable in the current sandbox (no network access to install
PyMuPDF). Written to the same contract as txt_parser/docx_parser so it is
a drop-in once run in the target environment (`pip install -r
requirements.txt`), and covered by a real assertion in
tests/test_ingestion.py once that dependency is present.
"""
from __future__ import annotations

import io

import fitz  # PyMuPDF

from src.utils.logging import get_logger

logger = get_logger("ingestion.pdf")


def parse_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF, page by page.

    Deliberately does NOT attempt OCR. If a page has no text layer (common
    for scanned/image-only PDFs), it simply contributes no text -- the
    caller (src/utils/validation.assess_extraction_quality) is responsible
    for detecting an overall-empty result and surfacing that as "this looks
    like a scanned PDF" rather than silently treating the candidate as
    having submitted a blank resume.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("Failed to open PDF: %s", exc)
        raise ValueError(f"Could not read this PDF file: {exc}") from exc

    page_texts = []
    for page in doc:
        page_texts.append(page.get_text("text"))
    doc.close()

    return "\n".join(page_texts)
