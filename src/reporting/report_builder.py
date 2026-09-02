"""
Report builder (downloadable analysis report).

Builds a Markdown report from an AnalysisResult. Kept as a plain string
builder (no UI/framework dependency) so it can be reused by the Streamlit
Report page, the FastAPI service, or a future CLI/batch export path.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.nlp.resume_quality import ResumeQualityReport
from src.recommendations.recommendation_engine import Recommendation


def build_markdown_report(analysis_result, quality: ResumeQualityReport, recommendations: list[Recommendation]) -> str:
    resume = analysis_result.resume
    job = analysis_result.job
    match = analysis_result.match
    breakdown = match["score_breakdown"]

    lines: list[str] = []
    lines.append("# HireLens AI — Resume Match Report")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    lines.append(f"**Candidate:** {resume.name or '(not detected)'}  ")
    lines.append(f"**Target role:** {job.title or '(not detected)'}")
    lines.append("")
    lines.append(f"## Match Score: {round(breakdown.final_score * 100)}%")
    lines.append("")
    lines.append("| Component | Score |")
    lines.append("|---|---|")
    lines.append(f"| Skills | {round(breakdown.skills_score * 100)}% |")
    lines.append(f"| Experience | {round(breakdown.experience_score * 100)}% |")
    lines.append(f"| Semantic Fit | {round(breakdown.semantic_score * 100)}% |")
    lines.append(f"| Projects | {round(breakdown.projects_score * 100)}% |")
    lines.append(f"| Education | {round(breakdown.education_score * 100)}% |")
    lines.append("")
    if breakdown.semantic_source == "tfidf_fallback":
        lines.append(
            "> Note: semantic similarity fell back to the TF-IDF baseline in this run "
            "(no embedding model available)."
        )
        lines.append("")

    lines.append("## Skill Comparison")
    lines.append("")
    lines.append("| Skill | Requirement | Status |")
    lines.append("|---|---|---|")
    for c in match["skill_comparison"]:
        extra = f" (via {c.related_skill_found})" if c.related_skill_found else ""
        lines.append(f"| {c.skill}{extra} | {c.requirement_level} | {c.status} |")
    lines.append("")

    lines.append("## Skill Gaps")
    lines.append("")
    if match["skill_gaps"]:
        for g in match["skill_gaps"]:
            lines.append(f"- **[{g.priority}]** {g.reason}")
    else:
        lines.append("_No skill gaps detected._")
    lines.append("")

    lines.append("## Resume Quality")
    lines.append("")
    lines.append(f"Overall quality score: **{round(quality.overall_score * 100)}%**")
    lines.append("")
    for dim in quality.dimensions:
        lines.append(f"- **{dim.name}**: {round(dim.score * 100)}% — {dim.detail}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    if recommendations:
        for i, r in enumerate(recommendations, start=1):
            lines.append(f"{i}. **[{r.priority}]** {r.text}")
    else:
        lines.append("_No specific recommendations._")
    lines.append("")

    lines.append("## Explanation")
    lines.append("")
    lines.append("**Positive evidence:**")
    for p in match["explanation"].positive_evidence:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("**Negative evidence:**")
    for n in match["explanation"].negative_evidence:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("---")
    lines.append(
        "_This report reflects automated analysis against the supplied job description. "
        "It is not a validated hiring decision tool and should be used as a starting point "
        "for self-review, not a final assessment._"
    )

    return "\n".join(lines)
