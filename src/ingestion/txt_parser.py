"""Plain-text ingestion. The simplest parser -- also the reference for what
'raw text extraction' means for the other two parsers."""
from __future__ import annotations


def parse_txt(file_bytes: bytes) -> str:
    """Decode a .txt file to a normalized unicode string.

    Tries utf-8 first (the overwhelming common case), falls back to latin-1
    (which never raises) rather than crashing on an unusual encoding, per
    spec Section 7 ("handle unusual encoding").
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    return text
