"""
FastAPI service (Step 19).

Exposes the same pipeline the Streamlit app uses (src/pipeline.py) over
HTTP, so HireLens AI's matching logic can be integrated into other
systems (an ATS, a Slack bot, a batch job) without going through the UI.

NOT executable in the current sandbox (no network to install fastapi/
uvicorn). Written against the same src/pipeline.analyze_resume_vs_job
entry point already exercised by the Streamlit app and the test suite, so
the only untested surface here is the HTTP/serialization layer itself.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.schemas import AnalyzeResponse, HealthResponse  # noqa: E402
from src.embeddings.encoder import is_available as embeddings_available  # noqa: E402
from src.evaluation.experiment_tracking import is_available as mlflow_available  # noqa: E402
from src.pipeline import analyze_resume_vs_job, get_taxonomy  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.validation import ValidationError  # noqa: E402

logger = get_logger("api")

app = FastAPI(
    title="HireLens AI API",
    description="Resume-to-job matching, skill gap analysis, and explainable scoring.",
    version="0.1.0",
)

# Permissive CORS for local development; tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    taxonomy = get_taxonomy()
    return HealthResponse(
        status="ok",
        taxonomy_skill_count=len(taxonomy.all_canonical_names()),
        embeddings_available=embeddings_available(),
        mlflow_available=mlflow_available(),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(..., description="Resume file (.pdf, .docx, or .txt)"),
    job_text: str = Form(None, description="Pasted job description text"),
    job_file: UploadFile = File(None, description="Job description file (.pdf, .docx, or .txt)"),
) -> AnalyzeResponse:
    if not job_text and job_file is None:
        raise HTTPException(status_code=400, detail="Provide either job_text or job_file.")

    resume_bytes = await resume.read()

    try:
        if job_file is not None:
            job_bytes = await job_file.read()
            result = analyze_resume_vs_job(
                resume_bytes=resume_bytes,
                resume_filename=resume.filename,
                job_text_or_bytes=job_bytes,
                job_filename=job_file.filename,
            )
        else:
            result = analyze_resume_vs_job(
                resume_bytes=resume_bytes,
                resume_filename=resume.filename,
                job_text_or_bytes=job_text,
                job_filename=None,
            )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # last-resort guard so the API never 500s opaquely
        logger.exception("Unexpected error during /analyze")
        raise HTTPException(status_code=500, detail=f"Internal error while analyzing: {exc}") from exc

    match = result.match
    return AnalyzeResponse(
        resume_filename=resume.filename,
        resume_name=result.resume.name,
        job_title=result.job.title,
        score_breakdown=match["score_breakdown"],
        skill_comparison=match["skill_comparison"],
        skill_gaps=match["skill_gaps"],
        explanation=match["explanation"],
        baseline_tfidf_similarity=match["baseline_tfidf_similarity"],
        resume_extraction_warning=result.resume_ingestion.quality.message,
        job_extraction_warning=result.job_ingestion.quality.message if result.job_ingestion else None,
    )
