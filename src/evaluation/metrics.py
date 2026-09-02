"""
Evaluation metrics (Step 17).

Computes only what can be honestly computed from a small, hand-labeled
bucket-style dataset (strong/moderate/weak) -- NOT precision/recall/F1
in the classical sense (that would require a much larger labeled set and
a fixed decision threshold validated independently of this same small
sample), and NOT a claim of "accuracy" against real hiring outcomes.

What IS computed:
  - bucket_accuracy: does the predicted score fall inside the expected
    score range for its labeled bucket?
  - mean_bucket_distance: how many bucket-widths off is a wrong
    prediction (0 for correct, 1 for one bucket off, 2 for two)
  - rank_correlation: Spearman correlation between predicted score and
    the bucket's ordinal rank (strong=2, moderate=1, weak=0) -- checks
    whether the scorer at least orders pairs consistently with the
    labels, even where the exact bucket boundary is missed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

_BUCKET_RANGES = {
    "weak": (0.0, 0.40),
    "moderate": (0.40, 0.70),
    "strong": (0.70, 1.0),
}
_BUCKET_RANK = {"weak": 0, "moderate": 1, "strong": 2}


@dataclass
class EvaluationMetrics:
    n_pairs: int
    bucket_accuracy: float
    mean_bucket_distance: float
    rank_correlation: float
    per_pair: List[Dict]


def _bucket_for_score(score: float) -> str:
    for bucket, (low, high) in _BUCKET_RANGES.items():
        if low <= score < high or (bucket == "strong" and score == 1.0):
            return bucket
    return "weak"


def _spearman(x: List[float], y: List[float]) -> float:
    """Minimal Spearman rank correlation with no scipy dependency.
    Handles ties by average rank.
    """
    def rank(values: List[float]) -> List[float]:
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    if len(x) < 2:
        return float("nan")

    rx, ry = rank(x), rank(y)
    n = len(x)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return float("nan")
    return cov / denom


def compute_metrics(predictions: List[Tuple[str, float, str]]) -> EvaluationMetrics:
    """`predictions` is a list of (pair_id, predicted_score, expected_bucket)."""
    per_pair = []
    correct = 0
    distances = []
    pred_scores = []
    expected_ranks = []

    for pair_id, score, expected_bucket in predictions:
        predicted_bucket = _bucket_for_score(score)
        is_correct = predicted_bucket == expected_bucket
        distance = abs(_BUCKET_RANK[predicted_bucket] - _BUCKET_RANK[expected_bucket])
        correct += int(is_correct)
        distances.append(distance)
        pred_scores.append(score)
        expected_ranks.append(_BUCKET_RANK[expected_bucket])

        per_pair.append(
            {
                "pair_id": pair_id,
                "predicted_score": round(score, 4),
                "predicted_bucket": predicted_bucket,
                "expected_bucket": expected_bucket,
                "correct": is_correct,
                "bucket_distance": distance,
            }
        )

    n = len(predictions)
    return EvaluationMetrics(
        n_pairs=n,
        bucket_accuracy=round(correct / n, 4) if n else float("nan"),
        mean_bucket_distance=round(sum(distances) / n, 4) if n else float("nan"),
        rank_correlation=round(_spearman(pred_scores, expected_ranks), 4) if n else float("nan"),
        per_pair=per_pair,
    )
