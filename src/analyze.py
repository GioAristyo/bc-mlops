"""L5: select a candidate using the predeclared precision constraint and recall."""
from __future__ import annotations

import mlflow

from .common import OUTPUT_DIR, configure_mlflow, save_json


def main() -> None:
    configure_mlflow()
    runs = mlflow.search_runs(
        experiment_names=["bc-malignant-detection"],
        filter_string=(
            "tags.candidate = 'true' and "
            "metrics.cv_precision_mean >= 0.90"
        ),
        order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"],
        max_results=100,
    )
    if runs.empty:
        raise RuntimeError(
            "Tidak ada kandidat yang memenuhi cv_precision_mean >= 0.90. "
            "Periksa MLflow UI dan hasil training sebelum melanjutkan."
        )
    required = ["run_id", "tags.mlflow.runName", "tags.model_uri", "metrics.cv_recall_mean", "metrics.cv_precision_mean"]
    available = [col for col in required if col in runs.columns]
    print(runs[available].head(10).to_string(index=False))
    best = runs.iloc[0]
    selection = {
        "run_id": str(best["run_id"]),
        "run_name": str(best.get("tags.mlflow.runName", "unknown")),
        "model_uri": str(best.get("tags.model_uri", "")),
        "cv_recall_mean": float(best["metrics.cv_recall_mean"]),
        "cv_recall_std": float(best.get("metrics.cv_recall_std", float("nan"))),
        "cv_precision_mean": float(best["metrics.cv_precision_mean"]),
        "cv_f1_mean": float(best.get("metrics.cv_f1_mean", float("nan"))),
        "cv_roc_auc_mean": float(best.get("metrics.cv_roc_auc_mean", float("nan"))),
        "selection_rule": "max cv_recall_mean subject to cv_precision_mean >= 0.90; tie-break cv_f1_mean",
    }
    if not selection["model_uri"] or selection["model_uri"] == "nan":
        raise RuntimeError("URI model tidak ditemukan pada tag tags.model_uri run terpilih.")
    save_json(OUTPUT_DIR / "selected_candidate.json", selection)
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    print("\nKandidat terpilih:")
    for key, value in selection.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
