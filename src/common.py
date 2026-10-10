"""Shared configuration and helpers for the experiment pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
EXPERIMENT_NAME = "bc-malignant-detection"
REGISTERED_MODEL_NAME = "bc-malignant-classifier"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
RANDOM_STATE = 42
MIN_PRECISION = 0.90


def configure_mlflow() -> None:
    """Point the client to the local tracking server and select the experiment."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow belum terpasang. Aktifkan environment bc-mlops dan jalankan `mamba install -c conda-forge mlflow`.") from exc
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
