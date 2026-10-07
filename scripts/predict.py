"""Write a submission from saved weights, without retraining.

A weights directory is what `python -m soc_baseline.train --artifacts-dir DIR`
leaves behind: model.joblib, metrics.json (feature set, benign weight) and,
when trained with --source-label-mask, source_label_mask.json. The released
weights (see README) use the same layout. `--verdict-rule` applies a rule
saved by `scripts/apply_verdict_rule.py --save-rule` (I13 in
docs/iteration_log.md), so no training data is needed.

model.joblib is a pickle: only load weights from a source you trust.

Usage:
  python scripts/predict.py --model-dir weights/timefree \
      --test-path data/valid_input.parquet --output res.csv \
      --verdict-rule weights/verdict_rule.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from soc_baseline.constants import ID_COLUMN, LABELS, PRED_COLUMN
from soc_baseline.data import read_parquet_frame
from soc_baseline.submission import validate_submission_frame
from soc_baseline.train import predict_parquet_to_submission
from soc_baseline.verdict_rule import apply_verdict_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/valid_input.parquet"))
    parser.add_argument("--output", type=Path, default=Path("res.csv"))
    parser.add_argument("--verdict-rule", type=Path, default=None, help="Rule JSON from apply_verdict_rule.py --save-rule.")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    metrics = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    mask_path = args.model_dir / "source_label_mask.json"
    mask = json.loads(mask_path.read_text(encoding="utf-8")) if mask_path.exists() else None
    model = joblib.load(args.model_dir / "model.joblib")
    feature_set = metrics.get("feature_set", "full")
    benign_weight = float(metrics.get("benign_weight", 1.0))

    rows = predict_parquet_to_submission(
        model, args.test_path, args.output, args.chunk_size, None,
        feature_set=feature_set, source_mask=mask, benign_weight=benign_weight,
    )

    rule = None
    if args.verdict_rule is not None:
        rule = json.loads(args.verdict_rule.read_text(encoding="utf-8"))
        test = read_parquet_frame(args.test_path, columns=[ID_COLUMN, "vendor_name", "message_sanitized"])
        submission = pd.read_csv(args.output, dtype={ID_COLUMN: str})
        test = test.assign(**{ID_COLUMN: test[ID_COLUMN].astype(str)}).set_index(ID_COLUMN).loc[submission[ID_COLUMN]]
        submission[PRED_COLUMN] = apply_verdict_rule(test, submission[PRED_COLUMN].to_numpy(), rule)
        submission.to_csv(args.output, index=False)

    expected_ids = read_parquet_frame(args.test_path, columns=[ID_COLUMN])[ID_COLUMN].astype(str)
    submission = pd.read_csv(args.output, dtype={ID_COLUMN: str})
    validate_submission_frame(submission, expected_ids, LABELS)
    print(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "feature_set": feature_set,
                "source_label_mask": mask is not None,
                "benign_weight": benign_weight,
                "verdict_rule": rule,
                "rows": rows,
                "label_distribution": submission[PRED_COLUMN].value_counts().to_dict(),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
