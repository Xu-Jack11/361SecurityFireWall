"""Check that a model's predictions and features carry no time information.

Two tests (I13 in docs/iteration_log.md):

1. Counterfactual: re-predict the test rows after moving every timestamp into
   the 2024-07-26 capture window, rewriting every digit in the message
   (years, epochs, clock times, sanitizer IDs) and renaming every month and
   weekday. A time-independent model must not change a single prediction.
2. Probe: within one fixed log format whose malicious rows span 2022-2024 in
   train.parquet, fit a classifier on the feature documents to predict the
   row's year. Accuracy near the majority-year rate means the documents hold
   no usable time signal; `content` is probed alongside for contrast.

Usage:
  python scripts/check_time_independence.py --model-dir artifacts/timefree
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from soc_baseline.constants import FEATURE_COLUMNS, ID_COLUMN, LABEL_COLUMN
from soc_baseline.data import read_parquet_frame
from soc_baseline.features import build_log_documents
from soc_baseline.source_mask import source_keys
from soc_baseline.train import _predict
from soc_baseline.verdict_rule import apply_verdict_rule, fit_verdict_rule

CAPTURE_WINDOW = pd.Timestamp("2024-07-27 12:00", tz="UTC").timestamp()
MONTH_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")
WEEKDAY_RE = re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b")
PROBE_FORMATS = {
    "asa_deny": r" Deny ",
    "meraki_flows": r" flows ",
}


def counterfactual(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shifted = frame.copy()
    shifted["timestamp"] = CAPTURE_WINDOW
    codes, uniques = pd.factorize(shifted["message_sanitized"].fillna("").astype(str))
    rewritten = []
    for text in uniques:
        text = MONTH_RE.sub("Jul", text)
        text = WEEKDAY_RE.sub("Fri", text)
        digits = rng.integers(0, 10, size=len(text)).astype(str)
        rewritten.append("".join(digits[i] if ch.isdigit() else ch for i, ch in enumerate(text)))
    shifted["message_sanitized"] = np.asarray(rewritten, dtype=object)[codes]
    return shifted


def run_counterfactual(
    model_dir: Path, feature_set: str, test_path: Path, late_sample: int, seed: int, rule: dict | None
) -> dict:
    model = joblib.load(model_dir / "model.joblib")
    mask_path = model_dir / "source_label_mask.json"
    mask = json.loads(mask_path.read_text(encoding="utf-8")) if mask_path.exists() else None
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    benign_weight = float(metrics.get("benign_weight", 1.0))

    test = read_parquet_frame(test_path, columns=[ID_COLUMN, *FEATURE_COLUMNS])
    cutoff = pd.Timestamp("2024-07-26 11:10", tz="UTC").timestamp()
    early = test[test["timestamp"] < cutoff]
    late = test[test["timestamp"] >= cutoff].sample(n=min(late_sample, len(test)), random_state=seed)
    rows = pd.concat([early, late])

    def predict(frame: pd.DataFrame) -> np.ndarray:
        docs = build_log_documents(frame, feature_set)
        predictions = np.asarray(_predict(model, docs, source_keys(frame), mask, benign_weight))
        return apply_verdict_rule(frame, predictions, rule) if rule else predictions

    original = predict(rows)
    shifted = predict(counterfactual(rows, seed))
    changed = original != shifted
    return {
        "rows": int(len(rows)),
        "early_rows": int(len(early)),
        "late_rows_sampled": int(len(late)),
        "changed_predictions": int(changed.sum()),
        "changed_early": int(changed[: len(early)].sum()),
        "original_distribution": pd.Series(original).value_counts().to_dict(),
        "counterfactual_distribution": pd.Series(shifted).value_counts().to_dict(),
    }


def run_probe(train_path: Path, feature_sets: list[str], max_rows: int, seed: int) -> dict:
    train = read_parquet_frame(train_path, columns=[*FEATURE_COLUMNS, LABEL_COLUMN])
    malicious = train[train[LABEL_COLUMN] == "malicious"]
    messages = malicious["message_sanitized"].fillna("")
    results = {}
    for name, pattern in PROBE_FORMATS.items():
        rows = malicious[messages.str.contains(pattern, regex=False)]
        rows = rows.sample(n=min(max_rows, len(rows)), random_state=seed)
        years = pd.to_datetime(rows["timestamp"], unit="s", utc=True).dt.year.to_numpy()
        entry = {
            "rows": int(len(rows)),
            "years": {int(k): int(v) for k, v in zip(*np.unique(years, return_counts=True))},
            "majority_rate": float(pd.Series(years).value_counts(normalize=True).iloc[0]),
        }
        for feature_set in feature_sets:
            docs = build_log_documents(rows, feature_set)
            x = TfidfVectorizer(
                lowercase=True, min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                token_pattern=r"(?u)\b[\w.\-/:=@+]+\b",
            ).fit_transform(docs)
            probe = LogisticRegression(max_iter=1000)
            entry[f"{feature_set}_year_accuracy"] = float(cross_val_score(probe, x, years, cv=5).mean())
            entry[f"{feature_set}_distinct_documents"] = int(docs.nunique())
        results[name] = entry
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/timefree"))
    parser.add_argument("--feature-set", default="timefree")
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/valid_input.parquet"))
    parser.add_argument("--late-sample", type=int, default=200_000)
    parser.add_argument("--probe-rows", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verdict-rule", action="store_true", help="Apply the train-derived verdict rule too.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rule = None
    if args.verdict_rule:
        train = read_parquet_frame(args.train_path, columns=["vendor_name", "message_sanitized", LABEL_COLUMN])
        rule = fit_verdict_rule(train, train[LABEL_COLUMN])
    result = {
        "model_dir": str(args.model_dir),
        "feature_set": args.feature_set,
        "verdict_rule": rule,
        "counterfactual": run_counterfactual(
            args.model_dir, args.feature_set, args.test_path, args.late_sample, args.seed, rule
        ),
        "probe": run_probe(args.train_path, ["content", args.feature_set], args.probe_rows, args.seed),
    }
    out = args.out or args.model_dir / "time_independence.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
