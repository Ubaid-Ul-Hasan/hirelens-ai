"""DOCX ingestion via python-docx.

Extracts paragraph text AND table cell text (resumes frequently use tables
for skill grids or two-column layouts, and skipping tables silently would
mean losing real content instead of failing loudly).
"""
from __future__ import annotations

import io

from docx import Document

from src.utils.logging import get_logger

logger = get_logger("ingestion.docx")


def parse_docx(file_bytes: bytes) -> str:
    """Extract raw text from a .docx file's paragraphs and tables, in
    document order as much as python-docx's API allows.
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:  # malformed .docx
        logger.warning("Failed to open DOCX: %s", exc)
        raise ValueError(f"Could not read this DOCX file: {exc}") from exc

    chunks: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            chunks.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                chunks.append(" | ".join(cells))

    return "\n".join(chunks)
