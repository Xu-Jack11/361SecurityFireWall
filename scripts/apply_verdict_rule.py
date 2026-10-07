"""Override a submission with the train-derived vendor-less "block verdict" rule.

Time-free replacement for scripts/apply_date_rule.py (I13 in
docs/iteration_log.md): learns from the training labels that vendor-less
records with a firewall block verdict are malicious, checks the cell's support
and purity, and forces that label on the matching test rows. See
src/soc_baseline/verdict_rule.py for what the rule does and does not mean.

Usage:
  python scripts/apply_verdict_rule.py --submission artifacts/timefree/res_timefree.csv \
      --output artifacts/timefree/res_verdict_rule.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soc_baseline.constants import ID_COLUMN, LABEL_COLUMN, LABELS, PRED_COLUMN
from soc_baseline.data import read_parquet_frame
from soc_baseline.submission import validate_submission_frame
from soc_baseline.verdict_rule import apply_verdict_rule, fit_verdict_rule, verdict_cells

RULE_COLUMNS = ["vendor_name", "message_sanitized"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/valid_input.parquet"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--min-purity", type=float, default=0.999)
    args = parser.parse_args()

    train = read_parquet_frame(args.train_path, columns=[*RULE_COLUMNS, LABEL_COLUMN])
    rule = fit_verdict_rule(train, train[LABEL_COLUMN], args.min_rows, args.min_purity)

    test = read_parquet_frame(args.test_path, columns=[ID_COLUMN, *RULE_COLUMNS])
    submission = pd.read_csv(args.submission, dtype={ID_COLUMN: str})
    test = test.assign(**{ID_COLUMN: test[ID_COLUMN].astype(str)}).set_index(ID_COLUMN).loc[submission[ID_COLUMN]]
    before = submission[PRED_COLUMN].to_numpy()
    submission[PRED_COLUMN] = apply_verdict_rule(test, before, rule)
    validate_submission_frame(submission, test.index.to_series(), LABELS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "rule": rule,
                "test_rows_in_cell": int((verdict_cells(test) == rule["cell"]).sum()),
                "changed_rows": int((before != submission[PRED_COLUMN].to_numpy()).sum()),
                "label_distribution": submission[PRED_COLUMN].value_counts().to_dict(),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
