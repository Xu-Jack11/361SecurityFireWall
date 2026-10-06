"""Generalization experiments for the SOC baseline.

Isolates two variables against the near-perfect random-holdout score:

  split axis    E1/E2 use a time-based split of the competition train set
                (train on the earliest 80%, validate on the latest 20%);
                E3/E4 train on the competition data and evaluate on the
                re-sanitized HF `latest` dump (token/IP mappings remapped).
  feature axis  E2/E4 normalize memorization-prone tokens (sanitization IDs,
                IP literals, absolute-month tokens) in the documents.

Usage:
  python scripts/generalization_test.py [--out artifacts/generalization]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from soc_baseline.constants import FEATURE_COLUMNS, LABEL_COLUMN
from soc_baseline.data import read_parquet_frame, stratified_sample
from soc_baseline.features import build_log_documents
from soc_baseline.gpu_modeling import TorchTfidfClassifier

TRAIN_PATH = "data/train.parquet"
LATEST_PATH = "data/external/witfoo-precinct6-signals-latest.parquet"

MAX_FEATURES = 120_000
MIN_DF = 3
RANDOM_STATE = 42
TRAIN_SAMPLE = 300_000
PREDICT_CHUNK = 500_000

_SANITIZED_ID_RE = re.compile(r"(?i)(?<![a-z])(fuser|user|org|cred|host)-\d+(?:-\d+)*")
_IP_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}(?:\.\d{1,3})?\b")
_MONTH_TOKEN_RE = re.compile(r"timestamp_month=-?\d+")
_MONTH_NAME_RE = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b")
_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")


def _ip_class(match: re.Match) -> str:
    first, second = int(match.group(1)), int(match.group(2))
    if first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168):
        return "iprfc1918"
    if first == 100 and 64 <= second <= 127:
        return "ipcgnat"
    if first == 127:
        return "iploopback"
    return "ippublic"


def normalize_documents(docs: pd.Series) -> pd.Series:
    """Collapse dataset-version-specific artifacts to invariant placeholder tokens."""

    def _normalize(text: str) -> str:
        text = _SANITIZED_ID_RE.sub(lambda m: m.group(1).lower() + "-id", text)
        text = _IP_RE.sub(_ip_class, text)
        text = _MONTH_TOKEN_RE.sub("timestamp_month=x", text)
        text = _MONTH_NAME_RE.sub("mon", text)
        text = _LONG_NUM_RE.sub("num", text)
        return text

    return docs.map(_normalize)


def make_model() -> TorchTfidfClassifier:
    return TorchTfidfClassifier(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        random_state=RANDOM_STATE,
        device="cuda",
        require_cuda=True,
    )


def predict_chunked(model: TorchTfidfClassifier, docs: pd.Series) -> np.ndarray:
    parts = [
        model.predict(docs.iloc[start : start + PREDICT_CHUNK])
        for start in range(0, len(docs), PREDICT_CHUNK)
    ]
    return np.concatenate(parts)


def run_experiment(
    name: str,
    train_docs: pd.Series,
    train_labels: pd.Series,
    val_docs: pd.Series,
    val_labels: pd.Series,
    normalize: bool,
) -> dict:
    started = time.time()
    if normalize:
        train_docs = normalize_documents(train_docs)
        val_docs = normalize_documents(val_docs)
    model = make_model()
    model.fit(train_docs, train_labels)
    predictions = predict_chunked(model, val_docs)
    report = classification_report(val_labels, predictions, output_dict=True, zero_division=0)
    result = {
        "experiment": name,
        "normalized_features": normalize,
        "train_rows": len(train_docs),
        "val_rows": len(val_docs),
        "macro_f1": f1_score(val_labels, predictions, average="macro"),
        "per_class": {
            label: {k: report[label][k] for k in ("precision", "recall", "f1-score", "support")}
            for label in sorted(val_labels.unique())
        },
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/generalization")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = list(FEATURE_COLUMNS) + [LABEL_COLUMN]
    print("loading competition train set...", flush=True)
    train_frame = read_parquet_frame(TRAIN_PATH, columns=columns)

    results = []

    # E1/E2: time-based split of the competition data.
    ordered = train_frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * 0.8)
    early, late = ordered.iloc[:cut], ordered.iloc[cut:]
    early_sample = stratified_sample(early, LABEL_COLUMN, TRAIN_SAMPLE, RANDOM_STATE)
    print(
        f"time split: fit on {len(early_sample)} of {len(early)} early rows, "
        f"validate on {len(late)} late rows "
        f"(val labels: {late[LABEL_COLUMN].value_counts().to_dict()})",
        flush=True,
    )
    early_docs = build_log_documents(early_sample)
    late_docs = build_log_documents(late)
    early_labels = early_sample[LABEL_COLUMN]
    late_labels = late[LABEL_COLUMN]
    results.append(run_experiment("E1_time_split_raw", early_docs, early_labels, late_docs, late_labels, False))
    results.append(run_experiment("E2_time_split_normalized", early_docs, early_labels, late_docs, late_labels, True))
    del ordered, early, late, early_docs, late_docs

    # E3/E4: train on competition data, evaluate on the re-sanitized HF latest dump.
    train_sample = stratified_sample(train_frame, LABEL_COLUMN, TRAIN_SAMPLE, RANDOM_STATE)
    train_docs = build_log_documents(train_sample)
    train_labels = train_sample[LABEL_COLUMN]
    del train_frame
    print("loading re-sanitized latest dump...", flush=True)
    latest_frame = read_parquet_frame(LATEST_PATH, columns=columns)
    latest_docs = build_log_documents(latest_frame)
    latest_labels = latest_frame[LABEL_COLUMN]
    del latest_frame
    results.append(run_experiment("E3_cross_version_raw", train_docs, train_labels, latest_docs, latest_labels, False))
    results.append(run_experiment("E4_cross_version_normalized", train_docs, train_labels, latest_docs, latest_labels, True))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}", flush=True)
    summary = pd.DataFrame(
        [
            {
                "experiment": r["experiment"],
                "macro_f1": round(r["macro_f1"], 5),
                "suspicious_f1": round(r["per_class"].get("suspicious", {}).get("f1-score", float("nan")), 5),
                "malicious_f1": round(r["per_class"].get("malicious", {}).get("f1-score", float("nan")), 5),
                "seconds": r["seconds"],
            }
            for r in results
        ]
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
