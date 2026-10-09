"""Score pipeline runs on valid. The only module, with ablation_leak.py, that reads valid labels.

python -m soc_hybrid.evaluate --runs final,final_w1,...  → artifacts/hybrid/runs/<name>/scores.json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import events, valid_answers
from .metrics import confusion_frame, evaluate, summary_row
from .pipeline import RUNS

KEYS = ["doc", "pipeline", "vendor_name", "product_name", "llm_message"]


def row_frame(run: str) -> pd.DataFrame:
    """Every valid event with its document's components and its true label."""

    docs = pd.read_parquet(RUNS / run / "docs.parquet")
    va = events("valid")
    # docs.parquet drops the long text columns; rebuild the join key from the doc order of valid_documents().
    from .pipeline import valid_documents

    keys = valid_documents()[KEYS]
    docs = docs.drop(columns=[column for column in KEYS if column in docs.columns])
    docs = pd.concat([keys.reset_index(drop=True), docs.reset_index(drop=True)], axis=1)
    rows = va[["event_id"] + KEYS].merge(docs, on=KEYS, how="left")
    truth = valid_answers()
    rows["truth"] = truth.reindex(rows["event_id"]).to_numpy()
    return rows.drop(columns=KEYS)


def score_run(run: str) -> dict:
    rows = row_frame(run)
    result = {"final": evaluate(rows["truth"], rows["final"]), "classifier_only": evaluate(rows["truth"], rows["clf"])}
    routed = rows[rows["routed"]]
    if len(routed):
        result["routed_rows_final"] = evaluate(routed["truth"], routed["final"])
        result["routed_rows_classifier"] = evaluate(routed["truth"], routed["clf"])
    (RUNS / run / "scores.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    table = []
    for run in args.runs.split(","):
        result = score_run(run)
        table.append(summary_row(run, result["final"]))
        table.append(summary_row(f"{run} (classifier only)", result["classifier_only"]))
        print(f"\n== {run}\n{confusion_frame(result['final']).to_string()}")
        if "routed_rows_final" in result:
            print("routed rows:", summary_row("final", result["routed_rows_final"]), "| classifier:",
                  summary_row("clf", result["routed_rows_classifier"]))
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
