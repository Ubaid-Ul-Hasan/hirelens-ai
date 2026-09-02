"""
HireLens AI - Skill Gap & Resume Quality page (Section 27-ish, Step 23).

No scoring/analysis logic lives here -- reads from session_state (set by
Home.py) and calls into src.nlp.resume_quality / src.recommendations for
display-ready structures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.cards import load_css, section_title, status_badge  # noqa: E402

st.set_page_config(page_title="Skill Gap | HireLens AI", page_icon="\U0001f3af", layout="wide")

try:
    load_css("app/styles/main.css")
except FileNotFoundError:
    pass

if "analysis_result" not in st.session_state:
    st.warning("No analysis found yet. Go to the **Home** page to upload a resume and job description.")
    st.stop()

result = st.session_state["analysis_result"]
match = result.match
comparisons = match["skill_comparison"]
gaps = match["skill_gaps"]

st.markdown("## Skill Gap Analysis")

required = [c for c in comparisons if c.requirement_level == "required"]
preferred = [c for c in comparisons if c.requirement_level == "preferred"]

col1, col2 = st.columns(2)
with col1:
    section_title(f"Required Skills ({len(required)})")
    if required:
        for c in required:
            extra = f" <span style='color:#9ca3af'>via {c.related_skill_found}</span>" if c.related_skill_found else ""
            st.markdown(f"{status_badge(c.status)} **{c.skill}**{extra}", unsafe_allow_html=True)
    else:
        st.caption("No required skills were detected in the job description.")

with col2:
    section_title(f"Preferred Skills ({len(preferred)})")
    if preferred:
        for c in preferred:
            extra = f" <span style='color:#9ca3af'>via {c.related_skill_found}</span>" if c.related_skill_found else ""
            st.markdown(f"{status_badge(c.status)} **{c.skill}**{extra}", unsafe_allow_html=True)
    else:
        st.caption("No preferred skills were detected in the job description.")

st.write("")
section_title("Prioritized Gaps")
if gaps:
    for g in gaps:
        st.markdown(f"**{g.priority}** \u2014 {g.skill} ({g.requirement_level}): {g.reason}")
else:
    st.success("No skill gaps detected against this job description.")

st.write("---")
st.markdown("## Resume Quality")
st.caption("Independent of this specific job — how well-constructed is the resume itself?")

quality = result.quality
qcol1, qcol2 = st.columns([1, 2])
with qcol1:
    st.metric("Overall Quality Score", f"{round(quality.overall_score * 100)}%")

with qcol2:
    for dim in quality.dimensions:
        pct = round(dim.score * 100)
        st.progress(dim.score, text=f"{dim.name}: {pct}% \u2014 {dim.detail}")

st.write("")
section_title("Recommendations")
recs = result.recommendations
if recs:
    for i, r in enumerate(recs, start=1):
        st.markdown(f"**{i:02d}. [{r.priority}]** {r.text}")
else:
    st.caption("No recommendations -- resume quality and skill coverage both look solid.")

st.write("")
section_title("Suggested Learning Roadmap")
roadmap = result.roadmap
if roadmap.items:
    for item in roadmap.items:
        st.markdown(f"**{item.week_label}** \u2014 *{item.skill}* ({item.priority}): {item.action}")
    st.caption(roadmap.note)
else:
    st.caption("No skill gaps to build a roadmap around -- nice work.")
