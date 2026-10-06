"""Prior calibration of the deployed baseline under test-set label shift.

The test set appears rebalanced relative to the training prior, so the
argmax decision rule of the balanced-weight model may sit at the wrong
operating point. This script:

1. streams `predict_proba` over the test parquet with the saved model,
2. estimates the test-set class priors with EM (Saerens et al., 2002),
   treating the balanced-weight model's implicit prior as uniform,
3. reweights the posteriors toward the estimated priors and re-decides,
4. writes a calibrated submission plus a summary of what changed.

Usage:
  python scripts/prior_calibration.py [--model artifacts/model.joblib]
      [--out artifacts/calibration]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from soc_baseline.constants import FEATURE_COLUMNS, ID_COLUMN
from soc_baseline.features import build_log_documents
from soc_baseline.submission import validate_submission_frame

TEST_PATH = "data/valid_input.parquet"
BASELINE_SUBMISSION = "res.csv"
CHUNK_SIZE = 200_000


def stream_probabilities(model, test_path: str) -> tuple[pd.Series, np.ndarray]:
    parquet = pq.ParquetFile(test_path)
    schema_columns = set(parquet.schema.names)
    columns = [ID_COLUMN] + [c for c in FEATURE_COLUMNS if c in schema_columns]
    ids, probas = [], []
    for batch in parquet.iter_batches(batch_size=CHUNK_SIZE, columns=columns):
        frame = batch.to_pandas()
        ids.append(frame[ID_COLUMN].astype(str))
        probas.append(model.predict_proba(build_log_documents(frame)).astype(np.float32))
        print(f"  scored {sum(len(i) for i in ids):,} rows", flush=True)
    return pd.concat(ids, ignore_index=True), np.vstack(probas)


def em_estimate_priors(
    probas: np.ndarray,
    model_prior: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (estimated test priors, adjusted posteriors, iterations)."""

    prior = probas.mean(axis=0).astype(np.float64)
    adjusted = probas.astype(np.float64)
    for iteration in range(1, max_iter + 1):
        weights = prior / model_prior
        adjusted = probas * weights
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new_prior = adjusted.mean(axis=0)
        shift = float(np.abs(new_prior - prior).max())
        prior = new_prior
        if shift < tol:
            break
    return prior, adjusted, iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--out", default="artifacts/calibration")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    classes = np.asarray(model.classes_)
    print(f"model classes: {classes.tolist()}", flush=True)

    print("streaming probabilities over test set...", flush=True)
    event_ids, probas = stream_probabilities(model, TEST_PATH)

    # class_weight-balanced training makes the model's implicit prior uniform.
    model_prior = np.full(len(classes), 1.0 / len(classes))
    raw_mean = probas.mean(axis=0)
    prior, adjusted, iterations = em_estimate_priors(probas, model_prior)

    raw_labels = classes[np.argmax(probas, axis=1)]
    calibrated_labels = classes[np.argmax(adjusted, axis=1)]

    submission = pd.DataFrame({ID_COLUMN: event_ids, "pred_label": calibrated_labels})
    validate_submission_frame(submission, event_ids)
    submission_path = out_dir / "res_calibrated.csv"
    submission.to_csv(submission_path, index=False)

    changed = raw_labels != calibrated_labels
    summary = {
        "model_path": args.model,
        "test_rows": int(len(event_ids)),
        "em_iterations": iterations,
        "mean_raw_probability": dict(zip(classes.tolist(), np.round(raw_mean, 6).tolist())),
        "estimated_test_prior": dict(zip(classes.tolist(), np.round(prior, 6).tolist())),
        "raw_label_distribution": pd.Series(raw_labels).value_counts().to_dict(),
        "calibrated_label_distribution": pd.Series(calibrated_labels).value_counts().to_dict(),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "submission_path": str(submission_path),
    }
    if changed.any():
        flips = pd.crosstab(
            pd.Series(raw_labels[changed], name="raw"),
            pd.Series(calibrated_labels[changed], name="calibrated"),
        )
        summary["flips"] = {f"{r}->{c}": int(flips.loc[r, c]) for r in flips.index for c in flips.columns if flips.loc[r, c]}

    baseline = Path(BASELINE_SUBMISSION)
    if baseline.exists():
        base = pd.read_csv(baseline, dtype=str)
        merged = base.merge(submission, on=ID_COLUMN, suffixes=("_baseline", "_calibrated"))
        summary["agreement_with_deployed_res_csv"] = float(
            (merged["pred_label_baseline"] == merged["pred_label_calibrated"]).mean()
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
