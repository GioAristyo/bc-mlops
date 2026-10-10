"""L6: evaluate the selected candidate on the locked test set once."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .common import OUTPUT_DIR, TRACKING_URI, configure_mlflow, read_json, save_json
from .data import load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-repeat", action="store_true", help="Allow re-evaluation for practice only; do not tune against test results.")
    args = parser.parse_args()
    configure_mlflow()
    train_df, test_df, data_info = load_splits()
    selection_path = OUTPUT_DIR / "selected_candidate.json"
    if not selection_path.exists():
        raise FileNotFoundError("Jalankan `python -m src.analyze` terlebih dahulu.")
    selection = read_json(selection_path)
    client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
    parent_run = client.get_run(selection["run_id"])
    if parent_run.data.tags.get("test_evaluated") == "true" and not args.allow_repeat:
        raise RuntimeError(
            "Test set sudah pernah dievaluasi untuk run ini. Demi menghindari test leakage, evaluasi dihentikan. "
            "Gunakan --allow-repeat hanya untuk latihan teknis, bukan untuk memilih model."
        )

    X_test = test_df.drop(columns="target")
    y_test = test_df["target"].astype(int)
    model = mlflow.sklearn.load_model(selection["model_uri"])
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        "n_test_samples": int(len(y_test)),
        "selected_run_id": selection["run_id"],
        "selected_run_name": selection["run_name"],
        "data_md5": str(data_info["data_md5"]),
        "test_split_md5": str(data_info["test_split_md5"]),
        "medical_disclaimer": "Educational benchmark only; not a medical diagnostic system.",
    }
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1], display_labels=["benign", "malignant"], cmap="Blues"
    )
    disp.figure_.tight_layout()
    plot_path = OUTPUT_DIR / "confusion_matrix_test.png"
    disp.figure_.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(disp.figure_)

    with mlflow.start_run(run_id=selection["run_id"]):
        
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["test_accuracy"],
                "test_recall": metrics["test_recall"],
                "test_precision": metrics["test_precision"],
                "test_f1": metrics["test_f1"],
                "test_roc_auc": metrics["test_roc_auc"],
            }
        )

        try:
            mlflow.log_input(
                mlflow.data.from_pandas(test_df, source="data/test.csv", name="bc-test", targets="target"),
                context="testing",
            )
        except (AttributeError, TypeError, mlflow.MlflowException) as exc:
            mlflow.set_tag("test_dataset_logging_warning", str(exc)[:200])
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.set_tag("test_evaluated", "true")
        mlflow.set_tag("test_data_md5", str(data_info["data_md5"]))
        mlflow.set_tag("test_split_md5", str(data_info["test_split_md5"]))
    save_json(OUTPUT_DIR / "test_metrics.json", metrics)
    print("Evaluasi test final (hanya untuk kandidat terpilih):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Confusion matrix tersimpan di: {plot_path}")


if __name__ == "__main__":
    main()
