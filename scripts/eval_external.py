"""Evaluate the time-dependent and time-free versions on samples of the external WitFoo dumps.

Datasets (see docs/public_datasets.md and I3/I10 in docs/iteration_log.md):
  v4           witfoo-precinct6-signals-v4: benign/suspicious rows are the
               competition train rows verbatim, malicious = train + 14,052
  latest       witfoo-precinct6-signals-latest: same events re-sanitized
               (new token namespace) and partly relabeled
  v2_live      precinct6-v2.1.0 signals: live capture 07-26..08-01, incident
               leads labeled malicious in place, vendor always attached
  v2_incident  precinct6-v2.1.0 incident_signals: historical leads, all malicious

Every group is also reported on its "novel" rows, whose raw message never
occurs in train.parquet (v4 is otherwise mostly in-sample).

Versions: B content model alone, C = B + source mask, D = C + date rule,
F_nomask timefree model alone, F = F_nomask + source mask, G = F + verdict
rule, G_bw10 = timefree (benign weight 10) + source mask + verdict rule.

Usage:
  python scripts/eval_external.py [--rows 300000] [--incident-rows 50000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from soc_baseline.constants import FEATURE_COLUMNS, LABEL_COLUMN, LABELS
from soc_baseline.data import read_parquet_frame, stratified_sample
from soc_baseline.features import build_log_documents
from soc_baseline.source_mask import source_keys
from soc_baseline.train import _predict
from soc_baseline.verdict_rule import apply_verdict_rule, fit_verdict_rule

EXTERNAL = Path("data/external")
DATASETS = {
    "v4": EXTERNAL / "witfoo-precinct6-signals-v4.parquet",
    "latest": EXTERNAL / "witfoo-precinct6-signals-latest.parquet",
    "v2_live": EXTERNAL / "precinct6-v2.1.0/signals.parquet",
    "v2_incident": EXTERNAL / "precinct6-v2.1.0/incident_signals.parquet",
}
MODELS = {
    "content": (Path("artifacts/content"), "content"),
    "timefree": (Path("artifacts/timefree"), "timefree"),
    "timefree_bw10": (Path("artifacts/timefree_bw10"), "timefree"),
}
RANDOM_STATE = 42


def load_model(directory: Path) -> tuple:
    model = joblib.load(directory / "model.joblib")
    mask = json.loads((directory / "source_label_mask.json").read_text(encoding="utf-8"))
    weight = float(json.loads((directory / "metrics.json").read_text(encoding="utf-8")).get("benign_weight", 1.0))
    return model, mask, weight


def predict_versions(frame: pd.DataFrame, models: dict, cutoff: float, rule: dict) -> dict[str, np.ndarray]:
    sources = source_keys(frame)
    raw = {}
    for name, (model, mask, weight, feature_set) in models.items():
        docs = build_log_documents(frame, feature_set)
        raw[name] = np.asarray(_predict(model, docs, sources, mask, weight))
        if weight == 1.0:
            raw[name + "_nomask"] = np.asarray(model.predict(docs))
    early = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy() < cutoff
    return {
        "B_content": raw["content_nomask"],
        "C_content_mask": raw["content"],
        "D_date_rule": np.where(early, "malicious", raw["content"]),
        "F_timefree_nomask": raw["timefree_nomask"],
        "F_timefree_mask": raw["timefree"],
        "G_verdict_rule": apply_verdict_rule(frame, raw["timefree"], rule),
        "G_verdict_rule_bw10": apply_verdict_rule(frame, raw["timefree_bw10"], rule),
    }


def score(labels: pd.Series, predictions: np.ndarray) -> dict:
    present = [label for label in LABELS if label in set(labels)]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=present, zero_division=0
    )
    return {
        "rows": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=present, average="macro", zero_division=0)),
        "errors": int((labels.to_numpy() != predictions).sum()),
        "per_class": {
            label: {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(s)}
            for label, p, r, f, s in zip(present, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(LABELS)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument("--incident-rows", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/external_eval"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = read_parquet_frame(args.train_path, columns=["timestamp", "vendor_name", "message_sanitized", LABEL_COLUMN])
    labels = train[LABEL_COLUMN].astype(str)
    seconds = pd.to_numeric(train["timestamp"], errors="coerce")
    cutoff = float(seconds[labels != "malicious"].min())
    rule = fit_verdict_rule(train, labels)
    train_messages = set(train["message_sanitized"].dropna())
    del train

    models = {name: (*load_model(directory), feature_set) for name, (directory, feature_set) in MODELS.items()}
    columns = list(FEATURE_COLUMNS) + [LABEL_COLUMN]
    results: dict = {"cutoff_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)), "verdict_rule": rule, "groups": {}}
    by_source = []
    for dataset, path in DATASETS.items():
        frame = read_parquet_frame(path, columns=columns)
        if dataset == "v2_incident":
            frame = frame.sample(n=min(args.incident_rows, len(frame)), random_state=RANDOM_STATE)
        else:
            frame = stratified_sample(frame, LABEL_COLUMN, args.rows, RANDOM_STATE)
        frame = frame.reset_index(drop=True)
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str)
        predictions = predict_versions(frame, models, cutoff, rule)

        subsets = {dataset: np.ones(len(frame), dtype=bool)}
        subsets[f"{dataset}/novel"] = ~frame["message_sanitized"].isin(train_messages).to_numpy()
        if dataset == "v2_live":
            day = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="s", utc=True).dt.strftime("%m-%d")
            subsets["v2_live/after_0726"] = (day != "07-26").to_numpy()
        for name, keep in subsets.items():
            truth = frame.loc[keep, LABEL_COLUMN]
            results["groups"][name] = {
                "labels": truth.value_counts().to_dict(),
                **{version: score(truth, pred[keep]) for version, pred in predictions.items()},
            }
            print(f"\n=== {name}  rows={int(keep.sum())}  labels={truth.value_counts().to_dict()}")
            for version in predictions:
                metrics = results["groups"][name][version]
                recalls = " ".join(f"{k[:3]}={v['recall']:.3f}" for k, v in metrics["per_class"].items())
                print(f"  {version:20s} macro_f1={metrics['macro_f1']:.4f} errors={metrics['errors']:7d}  recall {recalls}")

        is_mal = frame[LABEL_COLUMN].eq("malicious").to_numpy()
        table = pd.DataFrame({"dataset": dataset, "source": source_keys(frame).to_numpy()[is_mal]})
        for version, pred in predictions.items():
            table[version] = pred[is_mal] == "malicious"
        grouped = table.groupby(["dataset", "source"])
        summary = grouped.mean().round(3)
        summary.insert(0, "malicious_rows", grouped.size())
        by_source.append(summary.sort_values("malicious_rows", ascending=False))

    by_source = pd.concat(by_source)
    by_source.to_csv(args.out / "malicious_recall_by_source.csv")
    print(f"\nmalicious recall by source:\n{by_source.to_string()}")
    (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
