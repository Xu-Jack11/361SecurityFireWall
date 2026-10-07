"""Evaluate the saved models on a sample of WitFoo Precinct 6 v2.1.0.

v2.1.0 (HF tag `v2.1.0`, 2026-09-22) relabels the dataset so attacks are no
longer separated from benign traffic in time: `signals` is the live capture
(2024-07-26 11:10 -> 08-01) with 7,728 incident leads labeled malicious *in
place*, and `incident_signals` holds the historical leads (all malicious).
Tokens come from a rebuilt registry, so nothing is shared with train.parquet
except the underlying events of 2024-07-26.

Groups evaluated:
  live          stratified sample of `signals`
  live_after_0726  the part of that sample dated 07-27 or later (no overlap
                with the competition training capture)
  incident      random sample of `incident_signals` (malicious only -> recall)

Versions: A full-feature baseline, B content features, C = B + source mask,
D = C + train-derived date cutoff (before it => malicious).

Usage:
  python scripts/eval_precinct6_v2.py [--live-rows 300000] [--incident-rows 50000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from soc_baseline.constants import FEATURE_COLUMNS, LABEL_COLUMN, LABELS
from soc_baseline.data import read_parquet_frame, stratified_sample
from soc_baseline.features import build_log_documents
from soc_baseline.source_mask import predict_with_source_mask, source_keys

V2_DIR = Path("data/external/precinct6-v2.1.0")
BASELINE_MODEL = Path("artifacts/model.joblib")
CONTENT_DIR = Path("artifacts/content")
RANDOM_STATE = 42


def train_date_cutoff(train_path: Path) -> float:
    train = read_parquet_frame(train_path, columns=["timestamp", LABEL_COLUMN])
    seconds = pd.to_numeric(train["timestamp"], errors="coerce")
    labels = train[LABEL_COLUMN].astype(str)
    cutoff = float(seconds[labels != "malicious"].min())
    assert (labels[seconds < cutoff] == "malicious").all()
    return cutoff


def predict_versions(frame: pd.DataFrame, cutoff: float) -> dict[str, np.ndarray]:
    baseline = joblib.load(BASELINE_MODEL)
    content = joblib.load(CONTENT_DIR / "model.joblib")
    mask = json.loads((CONTENT_DIR / "source_label_mask.json").read_text())

    content_docs = build_log_documents(frame, "content")
    sources = source_keys(frame)
    masked = predict_with_source_mask(content, content_docs, sources, mask, list(content.classes_))
    early = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy() < cutoff
    return {
        "A_baseline": np.asarray(baseline.predict(build_log_documents(frame, "full"))),
        "B_content": np.asarray(content.predict(content_docs)),
        "C_content_mask": masked,
        "D_date_rule": np.where(early, "malicious", masked),
    }


def score(labels: pd.Series, predictions: np.ndarray) -> dict:
    present = [label for label in LABELS if label in set(labels)]
    report = classification_report(labels, predictions, labels=present, output_dict=True, zero_division=0)
    return {
        "rows": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=present, average="macro", zero_division=0)),
        "per_class": {label: {k: round(float(report[label][k]), 4) for k in ("precision", "recall", "f1-score", "support")} for label in present},
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(labels, predictions, labels=list(LABELS)).tolist(),
        },
        "predicted": pd.Series(predictions).value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }


def malicious_recall_by_source(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    is_mal = frame[LABEL_COLUMN].eq("malicious").to_numpy()
    sources = source_keys(frame).to_numpy()[is_mal]
    table = pd.DataFrame({"source": sources})
    for name, pred in predictions.items():
        table[name] = pred[is_mal] == "malicious"
    grouped = table.groupby("source")
    result = grouped.mean().round(3)
    result.insert(0, "malicious_rows", grouped.size())
    return result.sort_values("malicious_rows", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--live-rows", type=int, default=300_000)
    parser.add_argument("--incident-rows", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/precinct6_v2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    columns = list(FEATURE_COLUMNS) + [LABEL_COLUMN]
    cutoff = train_date_cutoff(args.train_path)

    live = stratified_sample(read_parquet_frame(V2_DIR / "signals.parquet", columns=columns), LABEL_COLUMN, args.live_rows, RANDOM_STATE)
    incident = read_parquet_frame(V2_DIR / "incident_signals.parquet", columns=columns)
    incident = incident.sample(n=min(args.incident_rows, len(incident)), random_state=RANDOM_STATE).reset_index(drop=True)

    results: dict = {"cutoff_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)), "groups": {}}
    pd.set_option("display.width", 220)
    for group, frame in (("live", live), ("incident", incident)):
        predictions = predict_versions(frame, cutoff)
        labels = frame[LABEL_COLUMN].astype(str)
        subsets = {group: np.ones(len(frame), dtype=bool)}
        if group == "live":
            day = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="s", utc=True).dt.strftime("%m-%d")
            subsets["live_after_0726"] = (day != "07-26").to_numpy()
        for name, keep in subsets.items():
            results["groups"][name] = {
                version: score(labels[keep], pred[keep]) for version, pred in predictions.items()
            }
            print(f"\n=== {name} ({int(keep.sum())} rows, labels {labels[keep].value_counts().to_dict()})")
            for version, metrics in results["groups"][name].items():
                recalls = {k: v["recall"] for k, v in metrics["per_class"].items()}
                print(f"  {version:15s} macro_f1={metrics['macro_f1']:.4f} recall={recalls} predicted={metrics['predicted']}")
        by_source = malicious_recall_by_source(frame, predictions)
        by_source.to_csv(args.out / f"{group}_malicious_recall_by_source.csv")
        print(f"\nmalicious recall by source ({group}):\n{by_source.to_string()}")

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
