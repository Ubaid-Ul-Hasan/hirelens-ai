"""
TF-IDF baseline matcher (Section 12) -- MANDATORY per spec.

This is the yardstick every "advanced" method (embeddings, hybrid) must be
evaluated against. Nothing here is allowed to silently improve without a
recorded before/after comparison (see src/evaluation).

We record vectorizer configuration, vocabulary size, and preprocessing
choices alongside the raw score, so a reported similarity number can
always be traced back to exactly how it was produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logging import get_logger

logger = get_logger("matching.tfidf")

_DEFAULT_VECTORIZER_PARAMS = dict(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True,
)


@dataclass
class TfidfBaselineResult:
    similarity: float
    vocabulary_size: int
    vectorizer_params: Dict = field(default_factory=dict)
    preprocessing: str = "lowercase, english stopword removal, unigrams+bigrams, sublinear TF"


def tfidf_similarity(resume_text: str, job_text: str) -> TfidfBaselineResult:
    """Compute cosine similarity between resume and job text using a
    TF-IDF vectorizer fit jointly on the two documents.

    NOTE: fitting on just this one resume/JD pair means IDF weighting is
    only meaningful relative to these two documents (df is 1 or 2 for
    every term). This is expected and documented behavior for a
    single-pair comparison; a corpus-level IDF (fit across many resumes/
    JDs) is a natural extension tracked in the evaluation notebooks, not a
    bug in this baseline.
    """
    if not resume_text.strip() or not job_text.strip():
        logger.warning("Empty resume or job text passed to TF-IDF baseline; returning 0 similarity.")
        return TfidfBaselineResult(similarity=0.0, vocabulary_size=0, vectorizer_params=_DEFAULT_VECTORIZER_PARAMS)

    vectorizer = TfidfVectorizer(**_DEFAULT_VECTORIZER_PARAMS)
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    except ValueError as exc:
        # e.g. vocabulary is empty after stopword removal (very short/odd input)
        logger.warning("TF-IDF vectorization failed: %s", exc)
        return TfidfBaselineResult(similarity=0.0, vocabulary_size=0, vectorizer_params=_DEFAULT_VECTORIZER_PARAMS)

    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return TfidfBaselineResult(
        similarity=float(sim),
        vocabulary_size=len(vectorizer.vocabulary_),
        vectorizer_params=_DEFAULT_VECTORIZER_PARAMS,
    )
