"""
Experiment runner (Step 17/18).

Runs every pair in data/evaluation/eval_pairs.json through:
  (a) the TF-IDF baseline alone
  (b) the hybrid matcher (src/matching/hybrid.py)
  (c) the product-facing final weighted score (src/scoring/scoring_engine.py)

...and reports bucket accuracy / rank correlation for each, so we can see
whether the more complex approaches actually outperform the mandatory
baseline on this (small, illustrative) set, rather than assuming they do.

Usage:
    python -m src.evaluation.run_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.experiment_tracking import is_available as mlflow_available
from src.evaluation.experiment_tracking import log_dict_artifact, log_metrics, log_params, start_run
from src.evaluation.metrics import compute_metrics
from src.matching.hybrid import compute_hybrid_match
from src.matching.tfidf import tfidf_similarity
from src.nlp.job_parser import parse_job_description
from src.nlp.resume_parser import parse_resume
from src.nlp.skill_extraction import SkillTaxonomy
from src.scoring.scoring_engine import run_full_match
from src.utils.config import EVAL_DATA_DIR
from src.utils.logging import get_logger

logger = get_logger("evaluation.run_experiment")


def load_eval_pairs():
    path = EVAL_DATA_DIR / "eval_pairs.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


def run_all(verbose: bool = True) -> dict:
    taxonomy = SkillTaxonomy()
    pairs = load_eval_pairs()

    baseline_preds, hybrid_preds, final_preds = [], [], []
    detail_rows = []

    for pair in pairs:
        resume = parse_resume(pair["resume"], taxonomy)
        job = parse_job_description(pair["job"], taxonomy)
        expected = pair["expected_bucket"]

        baseline = tfidf_similarity(resume.raw_text, job.raw_text).similarity
        hybrid = compute_hybrid_match(resume, job, taxonomy).hybrid_score
        final = run_full_match(resume, job, taxonomy)["score_breakdown"].final_score

        baseline_preds.append((pair["id"], baseline, expected))
        hybrid_preds.append((pair["id"], hybrid, expected))
        final_preds.append((pair["id"], final, expected))

        detail_rows.append(
            {
                "id": pair["id"],
                "expected_bucket": expected,
                "baseline_tfidf": round(baseline, 4),
                "hybrid": round(hybrid, 4),
                "final_score": round(final, 4),
            }
        )

    baseline_metrics = compute_metrics(baseline_preds)
    hybrid_metrics = compute_metrics(hybrid_preds)
    final_metrics = compute_metrics(final_preds)

    results = {
        "n_pairs": len(pairs),
        "mlflow_available": mlflow_available(),
        "baseline_tfidf": baseline_metrics.__dict__,
        "hybrid": hybrid_metrics.__dict__,
        "final_score": final_metrics.__dict__,
        "detail_rows": detail_rows,
    }

    with start_run(run_name="hirelens_eval_run"):
        log_params({"n_pairs": len(pairs), "dataset": "eval_pairs.json v0.1.0"})
        log_metrics(
            {
                "baseline_bucket_accuracy": baseline_metrics.bucket_accuracy,
                "hybrid_bucket_accuracy": hybrid_metrics.bucket_accuracy,
                "final_bucket_accuracy": final_metrics.bucket_accuracy,
                "baseline_rank_correlation": baseline_metrics.rank_correlation,
                "hybrid_rank_correlation": hybrid_metrics.rank_correlation,
                "final_rank_correlation": final_metrics.rank_correlation,
            }
        )
        log_dict_artifact(results, "eval_results.json")

    if verbose:
        print(f"\nEvaluated {len(pairs)} pairs (mlflow available: {mlflow_available()})\n")
        print(f"{'Pair':30} {'Expected':10} {'Baseline':10} {'Hybrid':10} {'Final':10}")
        for row in detail_rows:
            print(
                f"{row['id']:30} {row['expected_bucket']:10} "
                f"{row['baseline_tfidf']:<10} {row['hybrid']:<10} {row['final_score']:<10}"
            )
        print()
        print(f"{'Method':15} {'Bucket Acc':12} {'Mean Bucket Dist':18} {'Rank Corr'}")
        for name, m in [("Baseline TF-IDF", baseline_metrics), ("Hybrid", hybrid_metrics), ("Final Score", final_metrics)]:
            print(f"{name:15} {m.bucket_accuracy:<12} {m.mean_bucket_distance:<18} {m.rank_correlation}")

    return results


if __name__ == "__main__":
    run_all()
