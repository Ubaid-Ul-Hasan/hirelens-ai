"""
MLflow experiment tracking wrapper (Step 22).

Optional-import pattern: if mlflow isn't installed (e.g. this dev
sandbox), tracking calls become logged no-ops via a tiny stand-in context
manager rather than crashing the evaluation run. This means
`run_experiment.py` (Step 17/18) is runnable with or without mlflow
present -- exactly the same "usable even when an advanced component is
missing" principle applied to observability tooling rather than the ML
pipeline itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

from src.utils.logging import get_logger

logger = get_logger("evaluation.tracking")

try:
    import mlflow  # type: ignore

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


@contextmanager
def start_run(run_name: str):
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        logger.info("[mlflow unavailable] would start run: %s", run_name)
        yield


def log_params(params: Dict[str, Any]) -> None:
    if _MLFLOW_AVAILABLE:
        mlflow.log_params(params)
    else:
        logger.info("[mlflow unavailable] params: %s", params)


def log_metrics(metrics: Dict[str, float]) -> None:
    if _MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics)
    else:
        logger.info("[mlflow unavailable] metrics: %s", metrics)


def log_dict_artifact(data: Dict[str, Any], filename: str) -> None:
    if _MLFLOW_AVAILABLE:
        mlflow.log_dict(data, filename)
    else:
        logger.info("[mlflow unavailable] would log artifact: %s", filename)


def is_available() -> bool:
    return _MLFLOW_AVAILABLE
