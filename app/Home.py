"""
HireLens AI - Home / Upload page.

Per spec: keep business logic OUT of this file. This page only collects
input, calls src.pipeline.analyze_resume_vs_job, stores the result in
st.session_state, and navigates to the dashboard. All parsing/scoring
logic lives in src/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from app.components.cards import header_bar, load_css  # noqa: E402
from src.pipeline import analyze_resume_vs_job  # noqa: E402
from src.utils.validation import ValidationError  # noqa: E402

st.set_page_config(page_title="HireLens AI", page_icon="\U0001f4c4", layout="wide")

try:
    load_css()
except FileNotFoundError:
    pass  # CSS is cosmetic; app still functions without it

header_bar("HIRELENS AI", "AI Resume Intelligence Platform", status="")

st.markdown("### Analyze your resume against a target role")
st.caption(
    "Upload a resume and a job description. HireLens AI extracts skills, compares them "
    "against the role's requirements, and explains exactly why it produced its score."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### \U0001f4c4 Resume")
    resume_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], key="resume_upload")

with col2:
    st.markdown("#### \U0001f4bc Job Description")
    job_mode = st.radio("Provide the job description via:", ["Paste text", "Upload file"], horizontal=True)
    job_file = None
    job_text = None
    if job_mode == "Upload file":
        job_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], key="job_upload")
    else:
        job_text = st.text_area("Paste the job description", height=220, key="job_text_area")

st.write("")
analyze_clicked = st.button("ANALYZE RESUME", type="primary", use_container_width=False)

if analyze_clicked:
    if resume_file is None:
        st.error("Please upload a resume.")
    elif job_mode == "Upload file" and job_file is None:
        st.error("Please upload a job description file, or switch to 'Paste text'.")
    elif job_mode == "Paste text" and not (job_text and job_text.strip()):
        st.error("Please paste a job description, or switch to 'Upload file'.")
    else:
        with st.spinner("Parsing documents, extracting skills, and scoring the match..."):
            try:
                if job_mode == "Upload file":
                    result = analyze_resume_vs_job(
                        resume_bytes=resume_file.getvalue(),
                        resume_filename=resume_file.name,
                        job_text_or_bytes=job_file.getvalue(),
                        job_filename=job_file.name,
                    )
                else:
                    result = analyze_resume_vs_job(
                        resume_bytes=resume_file.getvalue(),
                        resume_filename=resume_file.name,
                        job_text_or_bytes=job_text,
                        job_filename=None,
                    )
            except ValidationError as e:
                st.error(str(e))
                st.stop()

        st.session_state["analysis_result"] = result

        if result.resume_ingestion.quality.message:
            st.warning(f"Resume: {result.resume_ingestion.quality.message}")
        if result.job_ingestion and result.job_ingestion.quality.message:
            st.warning(f"Job description: {result.job_ingestion.quality.message}")

        st.success("Analysis complete. Open **Resume Analysis** in the sidebar to view your dashboard.")

if "analysis_result" not in st.session_state:
    st.info("No analysis yet. Upload a resume and job description above, then click Analyze Resume.")
