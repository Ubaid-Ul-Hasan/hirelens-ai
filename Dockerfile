# HireLens AI - Streamlit dashboard image
FROM python:3.11-slim

WORKDIR /app

# System deps: PyMuPDF and some spaCy/torch wheels need these at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* the small spaCy model used by future NLP-assisted parsing
# (safe no-op if not yet wired into the pipeline)
RUN python -m spacy download en_core_web_sm || true

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
