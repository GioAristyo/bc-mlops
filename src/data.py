"""L1: prepare a fixed train/test split and record data provenance."""
from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .common import DATA_DIR, RANDOM_STATE, TEST_PATH, TRAIN_PATH, ensure_directories, save_json

TEST_SIZE = 0.20
METADATA_PATH = DATA_DIR / "metadata.json"


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint data values and row index for experiment lineage (not security)."""
    values = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.md5(values.tobytes()).hexdigest()


def split_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint a saved split independently from the full canonical dataset."""
    values = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.md5(values.tobytes()).hexdigest()


def prepare_data(force: bool = False) -> dict[str, object]:
    """Load sklearn's dataset, invert labels, save split and metadata.

    Original sklearn target names are ['malignant', 'benign'], with 0=malignant
    and 1=benign. This project flips the target so the positive label is 1=malignant.
    """
    ensure_directories()
    frame = load_breast_cancer(as_frame=True).frame.copy()
    frame["target"] = 1 - frame["target"].astype(int)
    data_md5 = dataset_fingerprint(frame)

    if force or not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        train_df, test_df = train_test_split(
            frame,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=frame["target"],
        )
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
    else:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        if len(train_df) + len(test_df) != len(frame):
            raise ValueError("Jumlah baris train/test tidak cocok dengan dataset asli. Jalankan python -m src.data.")

    # Re-read CSVs so split fingerprints reflect the exact persisted files and row order.
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    info: dict[str, object] = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "data_md5": data_md5,
        "train_split_md5": split_fingerprint(train_df),
        "test_split_md5": split_fingerprint(test_df),
        "n_samples": int(len(frame)),
        "n_features": int(frame.shape[1] - 1),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_size": TEST_SIZE,
        "split_seed": RANDOM_STATE,
        "positive_class": "malignant",
        "target_mapping": {"0": "benign", "1": "malignant"},
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
    }
    save_json(METADATA_PATH, info)

    print("Dataset berhasil disiapkan:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("Jumlah kelas train:", train_df["target"].value_counts().sort_index().to_dict())
    print("Jumlah kelas test :", test_df["target"].value_counts().sort_index().to_dict())
    print(f"Metadata disimpan : {METADATA_PATH}")
    return info


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return saved CSV splits and refresh provenance metadata."""
    if not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        prepare_data(force=True)
    # Recalculate metadata from the canonical dataset, while preserving the split files.
    frame = load_breast_cancer(as_frame=True).frame.copy()
    frame["target"] = 1 - frame["target"].astype(int)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    info: dict[str, object] = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "data_md5": dataset_fingerprint(frame),
        "train_split_md5": split_fingerprint(train_df),
        "test_split_md5": split_fingerprint(test_df),
        "n_samples": int(len(frame)),
        "n_features": int(frame.shape[1] - 1),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_size": TEST_SIZE,
        "split_seed": RANDOM_STATE,
        "positive_class": "malignant",
        "target_mapping": {"0": "benign", "1": "malignant"},
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
    }
    save_json(METADATA_PATH, info)
    return train_df, test_df, info


if __name__ == "__main__":
    prepare_data(force=True)
