from src.evaluation.metrics import compute_metrics
from src.matching.hybrid import compute_hybrid_match
from src.nlp.job_parser import parse_job_description
from src.nlp.resume_parser import parse_resume
from src.nlp.resume_quality import analyze_resume_quality
from src.nlp.skill_extraction import SkillTaxonomy
from src.nlp.weak_bullets import find_weak_bullets
from src.recommendations.recommendation_engine import build_recommendations
from src.recommendations.roadmap import build_roadmap
from src.scoring.scoring_engine import run_full_match


def test_hybrid_weights_always_sum_to_one():
    """Weights must sum to 1.0 regardless of whether semantic similarity
    is actually available in the running environment -- this should hold
    whether sentence-transformers is installed or not."""
    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython\n", tax)
    job = parse_job_description("Requirements\nPython\n", tax)
    result = compute_hybrid_match(resume, job, tax)
    # weights_used is rounded to 4dp for display, so allow for that rounding
    # error rather than requiring exact floating-point equality
    assert abs(sum(result.weights_used.values()) - 1.0) < 1e-3


def test_hybrid_weights_redistribute_when_semantic_forced_unavailable():
    """Deterministically exercises the redistribution branch by forcing
    semantic similarity unavailable, regardless of whether
    sentence-transformers happens to be installed in this environment --
    so this test doesn't silently stop testing the fallback path just
    because the dev machine has the optional dependency installed."""
    from unittest.mock import patch

    from src.matching.semantic import SemanticMatchResult

    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython\n", tax)
    job = parse_job_description("Requirements\nPython\n", tax)

    forced_unavailable = SemanticMatchResult(
        overall_similarity=None,
        section_results=[],
        available=False,
        unavailable_reason="forced unavailable for test",
    )

    with patch("src.matching.hybrid.compute_semantic_match", return_value=forced_unavailable):
        result = compute_hybrid_match(resume, job, tax)

    assert result.semantic_available is False
    assert result.weights_used["semantic_similarity"] == 0.0
    assert abs(sum(result.weights_used.values()) - 1.0) < 1e-3


def test_hybrid_score_is_bounded():
    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython, SQL, Docker\n", tax)
    job = parse_job_description("Requirements\nPython\nSQL\nDocker\n", tax)
    result = compute_hybrid_match(resume, job, tax)
    assert 0.0 <= result.hybrid_score <= 1.0


def test_weak_bullets_flags_vague_not_specific():
    bullets = [
        "Responsible for various tasks.",
        "Reduced latency by 40% using PyTorch and ONNX.",
        "Helped with stuff.",
        "Built a real-time fraud detection system serving 2M requests/day.",
    ]
    weak = find_weak_bullets(bullets)
    weak_texts = {w.text for w in weak}
    assert "Responsible for various tasks." in weak_texts
    assert "Helped with stuff." in weak_texts
    assert "Reduced latency by 40% using PyTorch and ONNX." not in weak_texts
    assert "Built a real-time fraud detection system serving 2M requests/day." not in weak_texts


def test_resume_quality_flags_missing_sections():
    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython\n", tax)  # no summary/experience/education
    report = analyze_resume_quality(resume)
    completeness = next(d for d in report.dimensions if d.name == "Section Completeness")
    assert completeness.score < 1.0
    assert "summary" in completeness.evidence or "education" in completeness.evidence


def test_resume_quality_full_resume_scores_higher_than_sparse_resume():
    tax = SkillTaxonomy()
    sparse = parse_resume("Skills\nPython\n", tax)
    full = parse_resume(
        "Jane Doe\njane@example.com | 555-1234\n\n"
        "Summary\nExperienced engineer.\n\n"
        "Experience\nML Engineer, Acme (2020-Present)\n"
        "Reduced latency by 40% using Python and Docker.\n\n"
        "Skills\nPython, Docker\n\n"
        "Education\nB.S. Computer Science\n",
        tax,
    )
    sparse_report = analyze_resume_quality(sparse)
    full_report = analyze_resume_quality(full)
    assert full_report.overall_score > sparse_report.overall_score


def test_recommendation_engine_includes_both_categories():
    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython\n", tax)
    job = parse_job_description("Requirements\nPython\nDocker\n", tax)
    match = run_full_match(resume, job, tax)
    quality = analyze_resume_quality(resume)
    recs = build_recommendations(match["skill_gaps"], quality)
    categories = {r.category for r in recs}
    assert "skill_gap" in categories
    assert "resume_quality" in categories
    # priorities must be in HIGH -> MEDIUM -> LOW order (non-decreasing rank)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ranks = [order[r.priority] for r in recs]
    assert ranks == sorted(ranks)


def test_roadmap_orders_high_priority_gaps_first():
    from src.schemas import SkillGapItem

    gaps = [
        SkillGapItem(skill="AWS", requirement_level="preferred", priority="LOW", reason="x"),
        SkillGapItem(skill="Docker", requirement_level="required", priority="HIGH", reason="x"),
    ]
    roadmap = build_roadmap(gaps)
    assert roadmap.items[0].skill == "Docker"
    assert roadmap.items[1].skill == "AWS"


def test_evaluation_metrics_perfect_predictions():
    predictions = [
        ("p1", 0.9, "strong"),
        ("p2", 0.5, "moderate"),
        ("p3", 0.1, "weak"),
    ]
    metrics = compute_metrics(predictions)
    assert metrics.bucket_accuracy == 1.0
    assert metrics.mean_bucket_distance == 0.0


def test_evaluation_metrics_detects_wrong_bucket():
    predictions = [
        ("p1", 0.1, "strong"),  # very wrong: predicted weak, labeled strong
    ]
    metrics = compute_metrics(predictions)
    assert metrics.bucket_accuracy == 0.0
    assert metrics.mean_bucket_distance == 2.0
