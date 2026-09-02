"""Plotly chart builders. All chart CONSTRUCTION lives here; pages just
call these functions and st.plotly_chart(...) the result."""
from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go


def skill_coverage_bar(skill_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of skill coverage percentages, sorted
    descending, matching the spec's ASCII mockup style."""
    items = sorted(skill_scores.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [round(v * 100) for _, v in items]

    colors = ["#22c55e" if v >= 70 else "#f59e0b" if v >= 40 else "#ef4444" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Coverage %"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(220, 34 * len(names)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def score_breakdown_bar(breakdown: Dict[str, float]) -> go.Figure:
    """'Why this score' horizontal bar chart across score components."""
    labels = {
        "skills_score": "Skills",
        "experience_score": "Experience",
        "semantic_score": "Semantic Fit",
        "projects_score": "Projects",
        "education_score": "Education",
    }
    names = [labels[k] for k in labels if k in breakdown]
    values = [round(breakdown[k] * 100) for k in labels if k in breakdown]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#6366f1",
            text=[f"{v}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def model_comparison_bar(model_scores: Dict[str, float]) -> go.Figure:
    """Model Lab: compare baseline TF-IDF vs semantic vs hybrid similarity."""
    names = list(model_scores.keys())
    values = [round(v * 100, 1) for v in model_scores.values()]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#6366f1", text=values, textposition="outside"))
    fig.update_layout(
        yaxis=dict(range=[0, 105], title="Similarity %"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig
