"""L8: send a small example to an MLflow model-serving REST endpoint."""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from .data import load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    _, test_df, _ = load_splits()
    X = test_df.drop(columns="target").head(args.rows)
    payload = {"dataframe_split": X.to_dict(orient="split")}
    response = requests.post(args.url, json=payload, timeout=30)
    response.raise_for_status()
    print("HTTP status:", response.status_code)
    print("Response:", response.json())
    print("Interpretasi output: 0=benign, 1=malignant. Ini bukan diagnosis medis.")


if __name__ == "__main__":
    main()
