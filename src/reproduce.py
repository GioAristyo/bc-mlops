"""L9: trace the champion alias to its MLflow run and verify the dataset fingerprint."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from .common import REGISTERED_MODEL_NAME, TRACKING_URI, configure_mlflow
from .data import load_splits


def main() -> None:
    configure_mlflow()
    _, _, data_info = load_splits()
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    run = client.get_run(version.run_id)
    tracked_hash = run.data.tags.get("data_md5")
    print("Model             :", REGISTERED_MODEL_NAME)
    print("Champion version  :", version.version)
    print("Run name / ID     :", run.data.tags.get("mlflow.runName"), run.info.run_id)
    print("Git commit        :", run.data.tags.get("mlflow.source.git.commit", "(tidak tersedia; belum ada commit saat training)"))
    print("Tracked data hash :", tracked_hash)
    print("Current data hash :", data_info["data_md5"])
    print("Tracked test split:", run.data.tags.get("test_split_md5"))
    print("Current test split:", data_info["test_split_md5"])
    print("Parameters        :", {k: v for k, v in run.data.params.items() if k.startswith("clf__") or k.startswith("best_")})
    print("Metrics           :", {k: round(v, 4) for k, v in run.data.metrics.items()})
    if tracked_hash != data_info["data_md5"]:
        raise AssertionError("Fingerprint dataset berbeda: jangan klaim data yang sama.")
    if run.data.tags.get("test_split_md5") != data_info["test_split_md5"]:
        raise AssertionError("Fingerprint test split berbeda: lineage test set tidak cocok.")
    print("PASS: dataset dan test split cocok; lineage dapat ditelusuri ke MLflow run.")


if __name__ == "__main__":
    main()
