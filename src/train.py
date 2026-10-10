"""L2-L4: baseline experiments, model selection, and SVC grid search."""
from __future__ import annotations

import json

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import OUTPUT_DIR, ROOT, configure_mlflow, ensure_directories, save_json
from .data import load_splits
from .tracking import CV, SCORING, run_experiment


def run_svc_grid_search(X_train, y_train, data_info: dict[str, object]) -> dict[str, str]:
    """Run 12 SVC configurations, logging one parent plus 12 child runs."""
    grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=42)),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=CV,
        scoring=SCORING,
        refit="recall",
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    with mlflow.start_run(
        run_name="10-svc-gridsearch",
        tags={
            "data_md5": str(data_info["data_md5"]),
            "train_split_md5": str(data_info["train_split_md5"]),
            "test_split_md5": str(data_info["test_split_md5"]),
            "split_seed": str(data_info["split_seed"]),
            "dataset": str(data_info["dataset"]),
            "stage": "tuning",
            "model_family": "svc",
            "candidate": "true",
        },
    ) as parent:
        mlflow.set_tag("mlflow.note.content", "H3: tuning SVC; seleksi CV refit berdasarkan recall.")
        mlflow.log_param("search_space", json.dumps(grid, sort_keys=True))
        mlflow.log_param("cv_n_splits", 5)
        try:
            dataset = mlflow.data.from_pandas(
                __import__("pandas").concat([X_train, y_train.rename("target")], axis=1),
                source="data/train.csv", name="bc-train", targets="target",
            )
            mlflow.log_input(dataset, context="training")
        except (AttributeError, TypeError, mlflow.MlflowException) as exc:
            mlflow.set_tag("dataset_logging_warning", str(exc)[:200])

        search.fit(X_train, y_train)
        result = search.cv_results_
        for index, params in enumerate(result["params"]):
            with mlflow.start_run(run_name=f"svc-trial-{index:02d}", nested=True):
                mlflow.set_tags({"stage": "tuning-trial", "model_family": "svc", "candidate": "false"})
                mlflow.log_params(params)
                mlflow.log_params({"trial_index": index})
                for metric in SCORING:
                    mlflow.log_metric(f"cv_{metric}_mean", float(result[f"mean_test_{metric}"][index]))
                    mlflow.log_metric(f"cv_{metric}_std", float(result[f"std_test_{metric}"][index]))

        best_index = int(search.best_index_)
        mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})
        for metric in SCORING:
            mlflow.log_metric(f"cv_{metric}_mean", float(result[f"mean_test_{metric}"][best_index]))
            mlflow.log_metric(f"cv_{metric}_std", float(result[f"std_test_{metric}"][best_index]))
        mlflow.log_metric("tuning_trials", int(len(result["params"])))
        mlflow.log_dict(grid, "search_space.json")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="environment")
        signature = infer_signature(X_train, search.best_estimator_.predict(X_train))
        info = mlflow.sklearn.log_model(
            search.best_estimator_, name="model", signature=signature, input_example=X_train.head(3)
        )
        model_uri = getattr(info, "model_uri", None) or f"runs:/{parent.info.run_id}/model"
        mlflow.set_tag("model_uri", model_uri)
        mlflow.set_tag("model_family", "svc")
        mlflow.set_tag("best_params", json.dumps(search.best_params_, sort_keys=True))
        print("SVC best params:", search.best_params_)
        print("SVC best CV recall:", float(result["mean_test_recall"][best_index]))
        return {"run_id": parent.info.run_id, "model_uri": model_uri, "run_name": "10-svc-gridsearch"}


def main() -> None:
    ensure_directories()
    configure_mlflow()
    train_df, _, data_info = load_splits()
    X_train = train_df.drop(columns="target")
    y_train = train_df["target"].astype(int)
    runs: list[dict[str, str]] = []

    # L2: baseline and controlled comparison. Only scaling changes in LR runs.
    runs.append(run_experiment(
        "00-dummy-most-frequent",
        Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        tags={"stage": "baseline", "model_family": "dummy"},
        note="Baseline batas bawah: selalu menebak kelas mayoritas.",
    ))
    runs.append(run_experiment(
        "01-logreg-noscale",
        Pipeline([("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H1: model linear tanpa standardisasi.",
    ))
    runs.append(run_experiment(
        "02-logreg-scaled",
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H2: hanya menambah StandardScaler; variabel lain tetap.",
    ))

    # L3: all candidates use the same scaler and identical CV folds.
    candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc": SVC(kernel="rbf", random_state=42),
    }
    for run_number, (family, classifier) in enumerate(candidates.items(), start=3):
        runs.append(run_experiment(
            f"{run_number:02d}-{family}-default",
            Pipeline([("scaler", StandardScaler()), ("clf", classifier)]),
            tags={"stage": "model-selection", "model_family": family},
            note="Baseline algoritma; preprocessing dan fold CV identik.",
        ))

    # L4: 12 SVC trials, as required by the scenario.
    runs.append(run_svc_grid_search(X_train, y_train, data_info))
    save_json(OUTPUT_DIR / "train_run_ids.json", {"runs": runs, "data": data_info})
    print(f"Selesai. Total parent/model candidate runs: {len(runs)}. Buka MLflow UI untuk membandingkan.")


if __name__ == "__main__":
    main()
