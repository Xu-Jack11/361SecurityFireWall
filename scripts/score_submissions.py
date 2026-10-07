"""Score submission CSVs against the labeled answer file for valid_input.

Usage:
  python scripts/score_submissions.py [--answer data/valid_answer_private.parquet] [NAME=PATH ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from soc_baseline.constants import ID_COLUMN, LABEL_COLUMN, LABELS, PRED_COLUMN

DEFAULT_SUBMISSIONS = {
    "A_baseline": "res.csv",
    "A_calibrated_I8": "artifacts/calibration/res_calibrated.csv",
    "A_witfoo_v4_I4": "artifacts/witfoo_v4/res.csv",
    "B_content": "artifacts/content/res_content.csv",
    "C_content_mask": "artifacts/content/res_source_mask.csv",
    "D_date_rule": "artifacts/content/res_date_rule.csv",
    "E_date_rule_bw10": "artifacts/content/res_date_rule_bw10.csv",
    "F0_timefree_no_action": "artifacts/timefree_ablation/res_timefree_no_action.csv",
    "F_timefree_mask": "artifacts/timefree/res_timefree.csv",
    "G_verdict_rule": "artifacts/timefree/res_verdict_rule.csv",
    "G_verdict_rule_bw10": "artifacts/timefree_bw10/res_verdict_rule_bw10.csv",
}


def score(answer: pd.Series, predicted: pd.Series) -> dict:
    report = classification_report(answer, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(answer, predicted)),
        "macro_f1": float(f1_score(answer, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(answer, predicted, labels=list(LABELS), average="weighted", zero_division=0)),
        "per_class": {
            label: {k: round(float(report[label][k]), 5) for k in ("precision", "recall", "f1-score")}
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(answer, predicted, labels=list(LABELS)).tolist(),
        "errors": int((answer != predicted).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer", type=Path, default=Path("data/valid_answer_private.parquet"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/answer_scores.json"))
    parser.add_argument("submissions", nargs="*", help="NAME=PATH pairs; defaults to every known version")
    args = parser.parse_args()

    submissions = dict(item.split("=", 1) for item in args.submissions) or DEFAULT_SUBMISSIONS
    answer = pd.read_parquet(args.answer, columns=[ID_COLUMN, LABEL_COLUMN]).set_index(ID_COLUMN)[LABEL_COLUMN].astype(str)

    results = {}
    for name, path in submissions.items():
        if not Path(path).exists():
            print(f"skip {name}: {path} not found")
            continue
        predicted = pd.read_csv(path, dtype={ID_COLUMN: str}).set_index(ID_COLUMN)[PRED_COLUMN]
        if len(predicted) != len(answer) or not predicted.index.isin(answer.index).all():
            raise ValueError(f"{path} does not cover exactly the answer's event_ids")
        results[name] = {"path": path, **score(answer, predicted.reindex(answer.index).astype(str))}

    print(f"{'version':18s} {'macro_f1':>9s} {'accuracy':>9s} {'errors':>8s}   per-class F1 (benign / suspicious / malicious)")
    for name, r in results.items():
        f1s = " / ".join(f"{r['per_class'][label]['f1-score']:.4f}" for label in LABELS)
        print(f"{name:18s} {r['macro_f1']:9.5f} {r['accuracy']:9.5f} {r['errors']:8d}   {f1s}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
