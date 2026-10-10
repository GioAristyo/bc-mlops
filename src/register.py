"""L7: register the selected model and promote it when the quality gate passes."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from .common import MIN_PRECISION, OUTPUT_DIR, REGISTERED_MODEL_NAME, TRACKING_URI, configure_mlflow, read_json


def champion_recall(client: MlflowClient) -> float:
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        run = client.get_run(champion.run_id)
        # Do not compare metrics from different test dataset versions.
        if run.data.tags.get("test_split_md5") is None:
            raise RuntimeError("Champion tidak memiliki test_split_md5; validasi lineage diperlukan.")
        return float(run.data.metrics["test_recall"])
    except mlflow.exceptions.MlflowException:
        return -1.0  # There is no champion yet; this is the initial registration.


def main() -> None:
    configure_mlflow()
    client = MlflowClient(tracking_uri=TRACKING_URI)
    selection = read_json(OUTPUT_DIR / "selected_candidate.json")
    metrics = read_json(OUTPUT_DIR / "test_metrics.json")
    if metrics["data_md5"] == "":
        raise RuntimeError("Fingerprint test kosong; menolak registrasi.")
    if metrics["test_precision"] < MIN_PRECISION:
        raise RuntimeError(
            f"Quality gate gagal: test precision={metrics['test_precision']:.3f} < {MIN_PRECISION:.2f}. "
            "Model tidak dipromosikan ke champion."
        )

    try:
        registered_model = client.get_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    version = mlflow.register_model(selection["model_uri"], REGISTERED_MODEL_NAME)
    client.update_model_version(
        REGISTERED_MODEL_NAME,
        version.version,
        description=(
            f"Dipilih melalui max CV recall dengan precision >= {MIN_PRECISION:.2f}; "
            f"test recall={metrics['test_recall']:.4f}, test precision={metrics['test_precision']:.4f}, "
            f"data_md5={metrics['data_md5']}"
        ),
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "data_md5", metrics["data_md5"])
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version.version)

    previous_recall = champion_recall(client)
    # Compare only when champion and challenger use the exact same test dataset.
    try:
        previous = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        previous_run = client.get_run(previous.run_id)
        if previous_run.data.tags.get("test_split_md5") != metrics["test_split_md5"]:
            raise RuntimeError("Quality gate menolak perbandingan: champion dan challenger memakai test split berbeda.")
    except mlflow.exceptions.MlflowException:
        previous_recall = -1.0

    margin = 0.005
    passed = metrics["test_precision"] >= MIN_PRECISION and metrics["test_recall"] >= previous_recall + margin
    status = "approved" if passed else "rejected"
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "validation_status", status)
    if passed:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
        print(f"Model version {version.version} dipromosikan menjadi @champion.")
    else:
        print(
            f"Model version {version.version} tetap @challenger (status rejected): "
            f"test recall {metrics['test_recall']:.4f} tidak lebih tinggi setidaknya {margin:.3f} "
            f"dari champion recall {previous_recall:.4f}."
        )
    print(f"Registry: {REGISTERED_MODEL_NAME} version {version.version}; status={status}")


if __name__ == "__main__":
    main()
