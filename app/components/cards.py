"""Reusable Streamlit UI fragments. Keeps app/pages/*.py free of raw HTML."""
from __future__ import annotations

import streamlit as st


def load_css(css_path: str = "app/styles/main.css") -> None:
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="hl-metric-card">
        <div class="hl-metric-value">{value}</div>
        <div class="hl-metric-label">{label}</div>
    </div>
    """


def render_metric_row(metrics: list[tuple[str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(metric_card(label, value), unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {"MATCHED": "hl-badge-matched", "PARTIAL": "hl-badge-partial", "MISSING": "hl-badge-missing"}.get(
        status, "hl-badge-missing"
    )
    symbol = {"MATCHED": "\u2713", "PARTIAL": "~", "MISSING": "\u26a0"}.get(status, "")
    return f'<span class="hl-badge {cls}">{symbol} {status}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="hl-section-title">{text}</div>', unsafe_allow_html=True)


def header_bar(title: str, subtitle: str, status: str = "") -> None:
    st.markdown(
        f"""
        <div class="hl-header-bar">
            <div>
                <div class="hl-title">{title}</div>
                <div class="hl-subtitle">{subtitle}</div>
            </div>
            <div class="hl-subtitle">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
