"""Override a submission with the train-derived "early date => malicious" rule.

In data/train.parquet every row stamped before the first non-malicious row
(2024-07-26 00:xx UTC) is malicious, and no benign/suspicious row precedes it.
The labeled public WitFoo v4/latest dumps show the same split, and the test
set's 14,052 rows before that cutoff are all deny/drop/reject firewall and
flow records. This script learns the cutoff from the training labels only and
forces test rows before it to `malicious`, leaving later rows untouched.

This deliberately uses the dataset's construction (malicious = historical
attack records, benign/suspicious = a later capture window); it is a
competition-specific rule, not a security signal, so keep it out of the model.

Usage:
  python scripts/apply_date_rule.py --submission artifacts/content/res_source_mask.csv \
      --output artifacts/content/res_date_rule.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soc_baseline.constants import ID_COLUMN, LABEL_COLUMN, LABELS, PRED_COLUMN
from soc_baseline.data import read_parquet_frame
from soc_baseline.submission import validate_submission_frame


def unix_seconds(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def learn_cutoff(train: pd.DataFrame) -> float:
    """Earliest non-malicious timestamp; checks that everything before it is malicious."""

    seconds = unix_seconds(train["timestamp"])
    labels = train[LABEL_COLUMN].astype(str)
    cutoff = float(seconds[labels != "malicious"].min())
    before = labels[seconds < cutoff]
    if before.empty or (before != "malicious").any():
        raise ValueError("training data has no pure-malicious window before the first non-malicious row")
    return cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/valid_input.parquet"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = read_parquet_frame(args.train_path, columns=["timestamp", LABEL_COLUMN])
    cutoff = learn_cutoff(train)
    malicious_before = int((unix_seconds(train["timestamp"]) < cutoff).sum())

    test = read_parquet_frame(args.test_path, columns=[ID_COLUMN, "timestamp"])
    early_ids = set(test.loc[unix_seconds(test["timestamp"]) < cutoff, ID_COLUMN].astype(str))

    submission = pd.read_csv(args.submission, dtype={ID_COLUMN: str})
    early = submission[ID_COLUMN].isin(early_ids)
    changed = early & (submission[PRED_COLUMN] != "malicious")
    submission.loc[early, PRED_COLUMN] = "malicious"
    validate_submission_frame(submission, test[ID_COLUMN].astype(str), LABELS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "cutoff_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)),
                "train_rows_before_cutoff_all_malicious": malicious_before,
                "test_rows_before_cutoff": int(early.sum()),
                "changed_rows": int(changed.sum()),
                "label_distribution": submission[PRED_COLUMN].value_counts().to_dict(),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
