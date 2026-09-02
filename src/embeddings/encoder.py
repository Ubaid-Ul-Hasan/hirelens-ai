"""
Sentence/document embedding encoder (Section 13, Step 8).

Wraps sentence-transformers behind a small interface so the rest of the
codebase never imports sentence_transformers directly. This is what makes
Engineering Rule #19 ("keep the app usable even if advanced components
fail") possible: every caller asks `is_available()` first and degrades
predictably (see src/matching/semantic.py and src/matching/hybrid.py)
instead of crashing when the model can't be loaded (no network, no disk
space, first-run download failure, etc).

NOT executable in the current sandbox (no network to install
sentence-transformers or download model weights). Written to be a
drop-in once run in the target environment.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np

from src.utils.config import CONFIG
from src.utils.logging import get_logger

logger = get_logger("embeddings.encoder")

_model = None
_load_attempted = False
_load_error: Optional[str] = None


def _try_load_model():
    global _model, _load_attempted, _load_error
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer  # local import by design

        _model = SentenceTransformer(CONFIG.embedding.model_name)
        logger.info("Loaded embedding model: %s", CONFIG.embedding.model_name)
    except Exception as exc:  # ImportError, OSError (no network/weights), etc.
        _load_error = str(exc)
        logger.warning(
            "Embedding model unavailable (%s). Semantic similarity will be "
            "unavailable; callers must fall back to the TF-IDF baseline.",
            exc,
        )
        _model = None
    return _model


def is_available() -> bool:
    """Cheap check callers use before attempting to encode anything."""
    return _try_load_model() is not None


def unavailability_reason() -> Optional[str]:
    return _load_error


def encode(texts: List[str]) -> Optional[np.ndarray]:
    """Encode a list of strings to embedding vectors.

    Returns None (not an exception) if the model could not be loaded, so
    callers can implement fallback behavior with a simple `if result is
    None` check rather than a try/except around every call site.
    """
    model = _try_load_model()
    if model is None:
        return None
    if not texts:
        return np.zeros((0, 384))  # matches all-MiniLM-L6-v2's dim; harmless if unused
    return model.encode(texts, batch_size=CONFIG.embedding.batch_size, show_progress_bar=False)


@lru_cache(maxsize=512)
def encode_single_cached(text: str) -> Optional[tuple]:
    """Cached single-string encode, returned as a tuple (hashable) so
    lru_cache works. Used for short, frequently repeated strings (e.g.
    canonical skill names) rather than whole documents.
    """
    result = encode([text])
    if result is None:
        return None
    return tuple(float(x) for x in result[0])


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
