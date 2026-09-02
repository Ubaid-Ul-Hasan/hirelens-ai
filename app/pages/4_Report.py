"""
HireLens AI - Report export page (Step 23).

Renders and offers for download the same Markdown report built by
src.reporting.report_builder, so what the user downloads and what's shown
on-screen are guaranteed to match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.cards import load_css  # noqa: E402
from src.reporting.report_builder import build_markdown_report  # noqa: E402

st.set_page_config(page_title="Report | HireLens AI", page_icon="\U0001f4c4", layout="wide")

try:
    load_css("app/styles/main.css")
except FileNotFoundError:
    pass

if "analysis_result" not in st.session_state:
    st.warning("No analysis found yet. Go to the **Home** page to upload a resume and job description.")
    st.stop()

result = st.session_state["analysis_result"]
report_md = build_markdown_report(result, result.quality, result.recommendations)

st.markdown("## Downloadable Report")
st.caption("The exact content below is what gets downloaded — nothing is added or removed on export.")

st.download_button(
    label="\u2b07\ufe0f Download Report (Markdown)",
    data=report_md,
    file_name=f"hirelens_report_{(result.resume.name or 'candidate').replace(' ', '_')}.md",
    mime="text/markdown",
)

st.write("---")
st.markdown(report_md)
