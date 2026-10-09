"""Train-internal holdout runs: every candidate classifier on every holdout of splits.py.

python -m soc_hybrid.holdout --classifier tfidf_word_lr [--weights doc] [--cap 20000] [--jobs 12]
  → artifacts/hybrid/holdout/<run>/probs.parquet (one row per held-out unique document)
    artifacts/hybrid/holdout/<run>/summary.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import LABELS, OUT, events, unique_documents
from .decision import benign_first
from .metrics import evaluate, summary_row
from .models import make_classifier, training_weights
from .splits import holdouts

HOLDOUT_DIR = OUT / "holdout"
BENIGN_WEIGHTS = (1, 2, 5, 10, 20, 50)


def cap_cells(frame: pd.DataFrame, cap: int | None, seed: int = 0) -> pd.DataFrame:
    """At most ``cap`` unique documents per (source, label) cell, so one huge cell does not swamp the rest."""

    if not cap:
        return frame
    parts = [cell if len(cell) <= cap else cell.sample(n=cap, random_state=seed)
             for _, cell in frame.groupby(["source", "label"], sort=False)]
    return pd.concat(parts)


def fit_predict(unique: pd.DataFrame, test_mask: np.ndarray, classifier: str, weights: str, cap: int | None,
                params: dict) -> np.ndarray:
    train = cap_cells(unique[~test_mask], cap)
    model = make_classifier(classifier, **params)
    model.fit(train["doc"].tolist(), train["label"].to_numpy(), training_weights(train, weights))
    return model.predict_proba(unique.loc[test_mask, "doc"].tolist())


def summarize(probs: pd.DataFrame) -> dict:
    summary = {}
    proba = probs[[f"p_{label}" for label in LABELS]].to_numpy()
    for kind, rows in probs.groupby("kind"):
        p = proba[rows.index]
        entry = {}
        for w in BENIGN_WEIGHTS:
            pred = benign_first(p, w)
            entry[f"w{w}"] = {
                "doc": summary_row(f"w{w}", evaluate(rows["label"], pred)),
                "row": summary_row(f"w{w}", evaluate(rows["label"], pred, rows["rows"])),
            }
        summary[kind] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="tfidf_word_lr")
    parser.add_argument("--weights", default="doc", choices=["doc", "log", "rows"])
    parser.add_argument("--cap", type=int, default=20_000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--kinds", default="ud5,cl5,loso,loco")
    parser.add_argument("--params", default="{}", help="JSON kwargs for the classifier")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    params = json.loads(args.params)
    name = args.name or f"{args.classifier}_{args.weights}_cap{args.cap}"
    out = HOLDOUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    unique = unique_documents(events("train"))
    kinds = set(args.kinds.split(","))
    plan = [(kind, holdout, mask) for kind, holdout, mask in holdouts(unique) if kind in kinds]
    began = time.time()
    jobs = 1 if args.classifier == "textcnn" and params.get("device", "cuda") != "cpu" else args.jobs
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(fit_predict)(unique, mask, args.classifier, args.weights, args.cap, params) for _, _, mask in plan
    )
    parts = []
    for (kind, holdout, mask), proba in zip(plan, results):
        part = unique.loc[mask, ["source", "label", "rows", "event_id"]].copy()
        part["doc_index"] = np.flatnonzero(mask)
        part["kind"], part["holdout"] = kind, holdout
        for i, label in enumerate(LABELS):
            part[f"p_{label}"] = proba[:, i]
        parts.append(part)
    probs = pd.concat(parts, ignore_index=True)
    probs.to_parquet(out / "probs.parquet", index=False)
    summary = {"classifier": args.classifier, "weights": args.weights, "cap": args.cap, "params": params,
               "seconds": round(time.time() - began), "kinds": summarize(probs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for kind, entry in summary["kinds"].items():
        for w in ("w1", "w10"):
            print(kind, w, "doc", entry[w]["doc"], "\n", " " * len(kind), w, "row", entry[w]["row"])
    print(f"{name}: {summary['seconds']}s")


if __name__ == "__main__":
    main()
