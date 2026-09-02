"""
HireLens AI - Model Lab (Section 16, Step 23).

Shows the same resume/job pair scored three different ways so the
difference between "mandatory baseline", "research-layer hybrid", and
"product-facing final score" is visible and explainable, rather than the
app presenting a single number as if it were the only way to measure fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.cards import load_css, section_title  # noqa: E402
from app.components.charts import model_comparison_bar  # noqa: E402
from src.embeddings.encoder import is_available as embeddings_available  # noqa: E402
from src.embeddings.encoder import unavailability_reason  # noqa: E402
from src.matching.hybrid import compute_hybrid_match  # noqa: E402
from src.pipeline import get_taxonomy  # noqa: E402

st.set_page_config(page_title="Model Lab | HireLens AI", page_icon="\U0001f52c", layout="wide")

try:
    load_css("app/styles/main.css")
except FileNotFoundError:
    pass

if "analysis_result" not in st.session_state:
    st.warning("No analysis found yet. Go to the **Home** page to upload a resume and job description.")
    st.stop()

result = st.session_state["analysis_result"]
match = result.match
taxonomy = get_taxonomy()

st.markdown("## Model Lab")
st.caption(
    "Compares three scoring approaches for the SAME resume/job pair currently loaded, "
    "so you can see what each method emphasizes rather than trusting one opaque number."
)

if not embeddings_available():
    st.info(
        f"\u2139\ufe0f Embedding-based semantic similarity is not available in this deployment "
        f"({unavailability_reason()}). All three scores below fall back to TF-IDF for the "
        f"semantic component -- this is disclosed in each score's breakdown, never hidden."
    )

hybrid_result = compute_hybrid_match(result.resume, result.job, taxonomy)

scores = {
    "TF-IDF Baseline": match["baseline_tfidf_similarity"],
    "Hybrid": hybrid_result.hybrid_score,
    "Final Score": match["score_breakdown"].final_score,
}

st.plotly_chart(model_comparison_bar(scores), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    section_title("TF-IDF Baseline")
    st.write(f"**{round(match['baseline_tfidf_similarity'] * 100, 1)}%** cosine similarity")
    st.caption("Lexical overlap only. The mandatory baseline every other method is measured against.")

with col2:
    section_title("Hybrid")
    st.write(f"**{round(hybrid_result.hybrid_score * 100, 1)}%**")
    st.caption("Lexical + semantic + required/preferred skill coverage + section relevance.")
    with st.expander("Component breakdown"):
        for k, v in hybrid_result.components.items():
            st.write(f"- {k}: {round(v * 100, 1)}%")
        st.write("Weights used (redistributed if semantic unavailable):")
        st.json(hybrid_result.weights_used)

with col3:
    section_title("Final Score")
    st.write(f"**{round(match['score_breakdown'].final_score * 100, 1)}%**")
    st.caption("Skills 35% / Experience 25% / Semantic 20% / Projects 10% / Education 10%.")
    with st.expander("Component breakdown"):
        b = match["score_breakdown"]
        st.write(f"- Skills: {round(b.skills_score * 100, 1)}%")
        st.write(f"- Experience: {round(b.experience_score * 100, 1)}%")
        st.write(f"- Semantic: {round(b.semantic_score * 100, 1)}%")
        st.write(f"- Projects: {round(b.projects_score * 100, 1)}%")
        st.write(f"- Education: {round(b.education_score * 100, 1)}%")

st.write("---")
st.markdown(
    "For a broader comparison across a labeled evaluation set (not just this one pair), "
    "run `python -m src.evaluation.run_experiment` — see `reports/error_analysis.md` for "
    "the most recent documented results."
)
