"""Diagnostics of a scored run on valid, for the report: where the classifier erred, what routing caught,
what the LLM changed.

python -m soc_hybrid.report --run final  → artifacts/hybrid/runs/<run>/diagnostics.json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .evaluate import row_frame
from .metrics import evaluate, summary_row
from .pipeline import RUNS


def diagnostics(run: str) -> dict:
    rows = row_frame(run)
    clf_err = rows["clf"] != rows["truth"]
    fin_err = rows["final"] != rows["truth"]
    routed = rows["routed"].astype(bool)
    changed = routed & (rows["final"] != rows["clf"])
    out = {
        "rows": int(len(rows)),
        "routed_rows": int(routed.sum()),
        "routed_docs": int(pd.read_parquet(RUNS / run / "docs.parquet")["routed"].sum()),
        "classifier_errors": int(clf_err.sum()),
        "classifier_errors_in_routed_rows": int((clf_err & routed).sum()),
        "final_errors": int(fin_err.sum()),
        "llm_changed_rows": int(changed.sum()),
        "llm_vetoed_alert_rows": int((changed & (rows["final"] == "benign")).sum()),
        "llm_vetoes_correct": int((changed & (rows["final"] == "benign") & (rows["truth"] == "benign")).sum()),
        "llm_raised_alert_rows": int((changed & (rows["clf"] == "benign")).sum()),
        "llm_raises_correct": int((changed & (rows["clf"] == "benign") & (rows["final"] == rows["truth"])).sum()),
        "llm_retyped_alert_rows": int((changed & (rows["clf"] != "benign") & (rows["final"] != "benign")).sum()),
        "fixed_rows": int((clf_err & ~fin_err).sum()),
        "broken_rows": int((~clf_err & fin_err).sum()),
    }
    by_source = rows.assign(clf_err=clf_err, fin_err=fin_err, routed=routed).groupby("source").agg(
        rows=("truth", "size"), routed=("routed", "sum"), clf_errors=("clf_err", "sum"), final_errors=("fin_err", "sum"))
    out["by_source"] = by_source[by_source[["routed", "clf_errors", "final_errors"]].sum(axis=1) > 0].astype(int).to_dict(orient="index")
    remaining = rows[fin_err].groupby(["source", "truth", "final"]).size().sort_values(ascending=False)
    out["remaining_errors"] = [{"source": s, "truth": t, "pred": p, "rows": int(n)} for (s, t, p), n in remaining.head(15).items()]
    out["final"] = summary_row("final", evaluate(rows["truth"], rows["final"]))
    out["classifier_only"] = summary_row("classifier", evaluate(rows["truth"], rows["clf"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="final")
    args = parser.parse_args()
    result = diagnostics(args.run)
    (RUNS / args.run / "diagnostics.json").write_text(json.dumps(result, indent=2, default=int), encoding="utf-8")
    print(json.dumps(result, indent=2, default=int, ensure_ascii=False))


if __name__ == "__main__":
    main()
