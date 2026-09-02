"""
HireLens AI - Resume Analysis dashboard (spec Section 26).

Reads the AnalysisResult stored in session_state by Home.py and renders
the main scorecard: overall match, skill coverage, strong matches vs
gaps, semantic fit, experience, and recommendations. No scoring or
parsing logic lives here -- only display.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.cards import load_css, render_metric_row, section_title, status_badge  # noqa: E402
from app.components.charts import score_breakdown_bar, skill_coverage_bar  # noqa: E402

st.set_page_config(page_title="Resume Analysis | HireLens AI", page_icon="\U0001f4c4", layout="wide")

try:
    load_css("app/styles/main.css")
except FileNotFoundError:
    pass

if "analysis_result" not in st.session_state:
    st.warning("No analysis found yet. Go to the **Home** page to upload a resume and job description.")
    st.stop()

result = st.session_state["analysis_result"]
match = result.match
breakdown = match["score_breakdown"]

final_pct = round(breakdown.final_score * 100)
if final_pct >= 75:
    verdict = "STRONG MATCH"
elif final_pct >= 50:
    verdict = "MODERATE MATCH"
else:
    verdict = "WEAK MATCH"

st.markdown(f"## Match Score: {final_pct}%  \u2014  {verdict}")
st.caption(
    "This score reflects alignment with the supplied job description, based on detected "
    "skills, experience signals, and text similarity. It is not a scientifically validated "
    "hiring probability."
)

comparisons = match["skill_comparison"]
required = [c for c in comparisons if c.requirement_level == "required"]
matched_required = [c for c in required if c.status == "MATCHED"]
gaps = match["skill_gaps"]

render_metric_row(
    [
        ("Match Score", f"{final_pct}%"),
        ("Skills", f"{round(breakdown.skills_score * 100)}%"),
        ("Experience", f"{round(breakdown.experience_score * 100)}%"),
        ("Skill Gaps", str(len(gaps))),
    ]
)

st.write("")
section_title("Skill Coverage")

skill_scores = {}
for c in comparisons:
    skill_scores[c.skill] = 1.0 if c.status == "MATCHED" else 0.5 if c.status == "PARTIAL" else 0.0
if skill_scores:
    st.plotly_chart(skill_coverage_bar(skill_scores), use_container_width=True)
else:
    st.info("No required or preferred skills were detected in the job description to compare against.")

col_a, col_b = st.columns(2)
with col_a:
    section_title("Strong Matches")
    matches = [c for c in comparisons if c.status == "MATCHED"]
    if matches:
        for c in matches:
            st.markdown(f"{status_badge(c.status)} **{c.skill}** ({c.requirement_level})", unsafe_allow_html=True)
    else:
        st.caption("No matched skills detected yet.")

with col_b:
    section_title("Skill Gaps")
    if gaps:
        for g in gaps:
            css_class = f"hl-gap-{g.priority.lower()}"
            st.markdown(
                f'<div class="{css_class}"><b>{g.priority}</b> &mdash; {g.skill} '
                f'<span style="color:#9ca3af;">({g.requirement_level})</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No skill gaps detected.")

st.write("")
section_title("Why This Score?")
st.plotly_chart(
    score_breakdown_bar(
        {
            "skills_score": breakdown.skills_score,
            "experience_score": breakdown.experience_score,
            "semantic_score": breakdown.semantic_score,
            "projects_score": breakdown.projects_score,
            "education_score": breakdown.education_score,
        }
    ),
    use_container_width=True,
)

if breakdown.semantic_source == "tfidf_fallback":
    st.caption(
        "\u2139\ufe0f Semantic (embedding-based) similarity was not available in this run; "
        "the semantic fit score above falls back to the TF-IDF baseline similarity."
    )

st.write("")
section_title("Explainability")
exp = match["explanation"]
ecol1, ecol2 = st.columns(2)
with ecol1:
    st.markdown("**Positive evidence**")
    for p in exp.positive_evidence:
        st.markdown(f"- \u2705 {p}")
with ecol2:
    st.markdown("**Negative evidence**")
    for n in exp.negative_evidence:
        st.markdown(f"- \u26a0\ufe0f {n}")

st.write("")
section_title("Recommendations")
if gaps:
    for i, g in enumerate(gaps[:5], start=1):
        st.markdown(f"**{i:02d}.** {g.reason}")
else:
    st.caption("No specific recommendations -- all detected requirements are covered.")
