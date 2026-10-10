"""Common MLflow experiment logger (used for baseline/model-selection runs)."""
from __future__ import annotations

import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from .common import ROOT
from .data import load_splits

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["recall", "precision", "f1", "roc_auc"]


def run_experiment(
    run_name: str,
    pipeline: Pipeline,
    tags: dict[str, str] | None = None,
    note: str = "",
) -> dict[str, str]:
    """Evaluate a pipeline with identical 5-fold CV and log the model + artifacts."""
    train_df, _, data_info = load_splits()
    X_train = train_df.drop(columns="target")
    y_train = train_df["target"].astype(int)
    started = time.perf_counter()

    with mlflow.start_run(run_name=run_name) as run:
        merged_tags = {
            "data_md5": str(data_info["data_md5"]),
            "train_split_md5": str(data_info["train_split_md5"]),
            "test_split_md5": str(data_info["test_split_md5"]),
            "split_seed": str(data_info["split_seed"]),
            "dataset": str(data_info["dataset"]),
            "stage": "model-selection",
            "candidate": "true",
            **(tags or {}),
        }
        mlflow.set_tags(merged_tags)
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_param("cv_n_splits", 5)
        mlflow.log_param("cv_seed", 42)
        mlflow.log_param("pipeline_steps", " -> ".join(pipeline.named_steps.keys()))
        mlflow.log_params({f"clf__{key}": value for key, value in pipeline[-1].get_params(deep=False).items()})

        # Dataset tracking is supported by MLflow 3. If this fails on a different
        # MLflow build, training metrics/model logging should still not be hidden.
        try:
            train_dataset = mlflow.data.from_pandas(
                train_df, source="data/train.csv", name="bc-train", targets="target"
            )
            mlflow.log_input(train_dataset, context="training")
        except (AttributeError, TypeError, mlflow.MlflowException) as exc:
            mlflow.set_tag("dataset_logging_warning", str(exc)[:200])

        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=-1,
            error_score="raise",
        )
        for metric in SCORING:
            values = cv_result[f"test_{metric}"]
            mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(values)))
            mlflow.log_metric(f"cv_{metric}_std", float(np.std(values, ddof=0)))
        mlflow.log_metric("fit_time_s", float(np.mean(cv_result["fit_time"])))
        mlflow.log_metric("total_run_time_s", float(time.perf_counter() - started))

        # Out-of-fold predictions diagnose training performance without test leakage.
        y_oof = cross_val_predict(pipeline, X_train, y_train, cv=CV, n_jobs=-1)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_train,
            y_oof,
            labels=[0, 1],
            display_labels=["benign", "malignant"],
            cmap="Blues",
        )
        disp.figure_.tight_layout()
        mlflow.log_figure(disp.figure_, "plots/confusion_matrix_oof.png")
        plt.close(disp.figure_)
        report = classification_report(
            y_train, y_oof, labels=[0, 1], target_names=["benign", "malignant"],
            zero_division=0, digits=4,
        )
        mlflow.log_text(report, "reports/classification_report_oof.txt")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="environment")

        pipeline.fit(X_train, y_train)
        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        log_model_kwargs = {}

        if pipeline[-1].__class__.__name__ in {
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
        }:
            log_model_kwargs["skops_trusted_types"] = [
                "sklearn.tree._tree.Tree"
            ]

        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            **log_model_kwargs,
        )

        model_uri = getattr(info, "model_uri", None) or f"runs:/{run.info.run_id}/model"
        mlflow.set_tag("model_uri", model_uri)
        mlflow.set_tag("model_family", merged_tags.get("model_family", "unknown"))
        print(
            f"{run_name}: recall={np.mean(cv_result['test_recall']):.4f}, "
            f"precision={np.mean(cv_result['test_precision']):.4f}, run_id={run.info.run_id}"
        )
        return {"run_id": run.info.run_id, "model_uri": model_uri, "run_name": run_name}
