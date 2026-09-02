"""
Central configuration for HireLens AI.

All tunable numbers (scoring weights, thresholds, model names) live here so
that:
  1. Nothing is hardcoded deep inside business logic.
  2. Weights can be justified / tuned / overridden without touching pipeline
     code (per spec rule: "keep all scoring weights configurable").

Values here are the *initial experimental* values proposed in the build
spec. They are NOT validated ground truth -- see src/evaluation for how
they should be checked against a labeled evaluation set before being
trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "evaluation"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TAXONOMY_PATH = PROJECT_ROOT / "src" / "nlp" / "skill_taxonomy.json"


# ---------------------------------------------------------------------------
# Document ingestion limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IngestionConfig:
    max_file_size_mb: int = 10
    allowed_extensions: tuple = (".pdf", ".docx", ".txt")
    min_extractable_chars: int = 50  # below this -> treat as empty/scanned


# ---------------------------------------------------------------------------
# Hybrid matching weights (Section 14 of spec)
# These combine into the "match" side of the pipeline, independent of the
# final weighted score below.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HybridMatchWeights:
    lexical_similarity: float = 0.20
    semantic_similarity: float = 0.30
    required_skill_match: float = 0.30
    preferred_skill_match: float = 0.10
    section_relevance: float = 0.10

    def as_dict(self) -> dict:
        return {
            "lexical_similarity": self.lexical_similarity,
            "semantic_similarity": self.semantic_similarity,
            "required_skill_match": self.required_skill_match,
            "preferred_skill_match": self.preferred_skill_match,
            "section_relevance": self.section_relevance,
        }


# ---------------------------------------------------------------------------
# Final match score weights (Section 16 of spec)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinalScoreWeights:
    skills: float = 0.35
    experience: float = 0.25
    semantic_fit: float = 0.20
    projects: float = 0.10
    education: float = 0.10

    def as_dict(self) -> dict:
        return {
            "skills": self.skills,
            "experience": self.experience,
            "semantic_fit": self.semantic_fit,
            "projects": self.projects,
            "education": self.education,
        }

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"FinalScoreWeights must sum to 1.0, got {total:.4f}"
            )


# ---------------------------------------------------------------------------
# Skill matching policy (Section 17)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkillMatchPolicy:
    # fuzzy alias match threshold (0-100, rapidfuzz scale) for near-duplicate
    # surface forms of the *same* skill (e.g. "Sklearn" vs "sci-kit learn")
    alias_fuzzy_threshold: int = 90
    # allow related-but-not-identical skills to count as a PARTIAL match
    # (e.g. TensorFlow <-> PyTorch under "Deep Learning Frameworks")
    allow_related_skill_partial_credit: bool = True
    partial_credit_weight: float = 0.5  # how much a PARTIAL match counts


# ---------------------------------------------------------------------------
# Embedding model config (used from Step 8 onward)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 16


@dataclass(frozen=True)
class AppConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    hybrid_weights: HybridMatchWeights = field(default_factory=HybridMatchWeights)
    final_weights: FinalScoreWeights = field(default_factory=FinalScoreWeights)
    skill_policy: SkillMatchPolicy = field(default_factory=SkillMatchPolicy)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


CONFIG = AppConfig()
CONFIG.final_weights.validate()
