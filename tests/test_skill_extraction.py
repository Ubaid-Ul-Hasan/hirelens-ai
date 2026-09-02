from src.nlp.skill_extraction import SkillTaxonomy, extract_skills


def test_taxonomy_normalizes_aliases():
    tax = SkillTaxonomy()
    assert tax.normalize("sklearn") == "Scikit-learn"
    assert tax.normalize("k8s") == "Kubernetes"
    assert tax.normalize("postgres") == "PostgreSQL"
    assert tax.normalize("js") == "JavaScript"
    assert tax.normalize("totally made up skill") is None


def test_tf_idf_does_not_false_trigger_tensorflow():
    """Regression test: 'TF-IDF' must not be detected as TensorFlow just
    because 'tf' looks like a short alias substring."""
    tax = SkillTaxonomy()
    detections = extract_skills("Built a TF-IDF based matching baseline.", tax)
    skill_names = {d.skill for d in detections}
    assert "TensorFlow" not in skill_names


def test_passing_mention_gets_lower_confidence():
    """'Worked with data stored in AWS' should NOT be treated as strong
    evidence of AWS expertise -- confidence should be reduced."""
    tax = SkillTaxonomy()
    detections = extract_skills("Worked with data stored in AWS S3 for pipelines.", tax)
    aws = next(d for d in detections if d.skill == "AWS")
    assert aws.confidence < 1.0


def test_direct_skill_list_gets_full_confidence():
    tax = SkillTaxonomy()
    detections = extract_skills("Skills: Python, AWS, Docker", tax)
    aws = next(d for d in detections if d.skill == "AWS")
    assert aws.confidence == 1.0


def test_evidence_snippets_are_captured():
    tax = SkillTaxonomy()
    detections = extract_skills("Built models using PyTorch for image classification.", tax)
    pytorch = next(d for d in detections if d.skill == "PyTorch")
    assert len(pytorch.evidence_snippets) > 0
    assert "PyTorch" in pytorch.evidence_snippets[0]
