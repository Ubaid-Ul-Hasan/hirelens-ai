from src.nlp.job_parser import parse_job_description
from src.nlp.resume_parser import parse_resume
from src.nlp.section_parser import parse_sections
from src.nlp.skill_extraction import SkillTaxonomy
from src.matching.tfidf import tfidf_similarity
from src.scoring.scoring_engine import run_full_match


def test_section_heading_synonyms_map_to_same_canonical_section():
    resume = "Work Experience\nDid stuff.\n\nEducation\nDegree stuff."
    parsed = parse_sections(resume)
    assert "experience" in parsed.sections

    resume2 = "Professional Experience\nDid stuff.\n\nEducation\nDegree stuff."
    parsed2 = parse_sections(resume2)
    assert "experience" in parsed2.sections
    assert parsed.sections["experience"] == parsed2.sections["experience"]


def test_contact_block_precedes_first_heading():
    resume = "Jane Doe\njane@example.com\n\nSkills\nPython"
    parsed = parse_sections(resume)
    assert "Jane Doe" in parsed.contact_block
    assert "jane@example.com" in parsed.contact_block


def test_job_parser_separates_required_and_preferred():
    tax = SkillTaxonomy()
    jd = (
        "Requirements\nPython\nSQL\nFamiliarity with Kubernetes is a plus\n\n"
        "Preferred\nAWS\nMLflow\n"
    )
    job = parse_job_description(jd, tax)
    assert "Python" in job.required_skills
    assert "SQL" in job.required_skills
    # inline "is a plus" should move Kubernetes to preferred even though
    # it's physically under the Requirements heading
    assert "Kubernetes" in job.preferred_skills
    assert "Kubernetes" not in job.required_skills
    assert "AWS" in job.preferred_skills
    assert "MLflow" in job.preferred_skills


def test_tfidf_baseline_ranks_relevant_job_higher():
    resume = "Senior ML engineer with Python, PyTorch, and Docker experience."
    close_job = "Seeking senior machine learning engineer with Python and PyTorch skills."
    far_job = "Seeking a marketing coordinator for social media campaigns."
    close = tfidf_similarity(resume, close_job)
    far = tfidf_similarity(resume, far_job)
    assert close.similarity > far.similarity


def test_job_title_not_guessed_from_heading_line():
    """Regression test: a JD with no title line, starting directly with a
    'Requirements' heading, must not report 'Requirements' as the title."""
    tax = SkillTaxonomy()
    job = parse_job_description("Requirements\nPython\nSQL\n", tax)
    assert job.title is None

    job_with_title = parse_job_description("Senior ML Engineer\n\nRequirements\nPython\n", tax)
    assert job_with_title.title == "Senior ML Engineer"


def test_full_pipeline_produces_sane_score_and_gaps():
    tax = SkillTaxonomy()
    resume_text = (
        "Jane Doe\njane@example.com\n\n"
        "Skills\nPython, SQL, Machine Learning\n\n"
        "Experience\nML Engineer, Acme (2020-Present)\nBuilt models with Python.\n"
    )
    jd_text = "Requirements\nPython\nSQL\nDocker\n\nPreferred\nAWS\n"

    resume = parse_resume(resume_text, tax)
    job = parse_job_description(jd_text, tax)
    result = run_full_match(resume, job, tax)

    breakdown = result["score_breakdown"]
    assert 0.0 <= breakdown.final_score <= 1.0

    gap_skills = {g.skill for g in result["skill_gaps"]}
    assert "Docker" in gap_skills  # missing required -> must appear as a gap
    high_priority = [g for g in result["skill_gaps"] if g.priority == "HIGH"]
    assert any(g.skill == "Docker" for g in high_priority)


def test_missing_skill_language_is_careful_not_absolute():
    """Per spec Section 19: never claim a candidate 'has no X experience' --
    only that X 'was not detected'."""
    tax = SkillTaxonomy()
    resume = parse_resume("Skills\nPython\n", tax)
    job = parse_job_description("Requirements\nPython\nDocker\n", tax)
    result = run_full_match(resume, job, tax)
    negative = " ".join(result["explanation"].negative_evidence)
    assert "not detected" in negative
    assert "has no" not in negative.lower()
