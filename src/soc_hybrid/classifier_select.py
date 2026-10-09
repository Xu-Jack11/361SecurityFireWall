"""Iteration 4: choose the classifier at the Bayes benign weight (w = 10) on train holdouts, ensembles included.

Candidates are the holdout runs and the equal-weight average of TextCNN with each TF-IDF variant. Each is
scored without the LLM on cl5 (in-distribution) and on LOSO/LOCO (out of distribution), with the same
minimax rule as route_select.py: 0.99 x cl5 cost + 0.01 x OOD cost, OOD aggregated per holdout (macro) and per
document (pooled), and the candidate with the smallest worst-case ratio to the best wins. The cross-fitted
temperature-scaled log-loss on cl5 is reported alongside (iteration 3 chose by it alone).

python -m soc_hybrid.classifier_select  → artifacts/hybrid/classifier_select.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .data import LABELS, OUT
from .holdout import HOLDOUT_DIR
from .metrics import BENIGN_MISS_COST
from .route_select import PRIOR_OOD

CNN = "textcnn_rows_cap20000"
TFIDF = ("tfidf_word_lr_doc_cap20000", "tfidf_word_lr_log_cap20000", "tfidf_word_lr_rows_cap20000")


def _load(run: str) -> pd.DataFrame:
    return pd.read_parquet(HOLDOUT_DIR / run / "probs.parquet").sort_values(["kind", "holdout", "doc_index"]).reset_index(drop=True)


def _temper(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _log_loss(proba, y, weight) -> float:
    return float(-(weight * np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1))).sum() / weight.sum())


def calibrated_log_loss(proba: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Class-balanced log-loss after a temperature fitted on the other folds (cross-fitted)."""

    class_weight = {k: len(y) / (len(LABELS) * max((y == k).sum(), 1)) for k in range(len(LABELS))}
    weight = np.array([class_weight[k] for k in y])
    scored = []
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        temperature = minimize_scalar(lambda t: _log_loss(_temper(proba[train], t), y[train], weight[train]),
                                      bounds=(0.05, 20), method="bounded").x
        scored.append((_temper(proba[test], temperature), y[test], weight[test]))
    return _log_loss(np.vstack([s[0] for s in scored]), np.concatenate([s[1] for s in scored]),
                     np.concatenate([s[2] for s in scored]))


def costs(proba: np.ndarray, y: np.ndarray, kinds: np.ndarray, holdouts: np.ndarray, benign_weight: float) -> dict:
    scores = proba.copy()
    scores[:, 0] *= benign_weight
    pred = scores.argmax(axis=1)
    cost = (pred != y).astype(float) + (BENIGN_MISS_COST - 1) * ((y == 0) & (pred != 0))
    out = {}
    for kind in ("cl5", "loso", "loco"):
        mask = kinds == kind
        out[f"{kind}_pooled"] = float(cost[mask].mean())
        out[f"{kind}_macro"] = float(pd.Series(cost[mask]).groupby(holdouts[mask]).mean().mean())
    return out


def main(benign_weight: float = 10.0) -> pd.DataFrame:
    base = _load(CNN)
    y = base["label"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    kinds, holdouts = base["kind"].to_numpy(), base["holdout"].to_numpy()
    probs = {}
    for run in (CNN, *TFIDF):
        frame = _load(run)
        assert (frame["doc_index"].to_numpy() == base["doc_index"].to_numpy()).all()
        probs[run] = frame[[f"p_{label}" for label in LABELS]].to_numpy()
    candidates = dict(probs)
    for run in TFIDF:
        candidates[f"mean({CNN}, {run})"] = (probs[CNN] + probs[run]) / 2
    cl5 = kinds == "cl5"
    rows = []
    for name, proba in candidates.items():
        row = {"classifier": name, "cl5_calibrated_log_loss": calibrated_log_loss(proba[cl5], y[cl5], holdouts[cl5]),
               **costs(proba, y, kinds, holdouts, benign_weight)}
        rows.append(row)
    table = pd.DataFrame(rows)
    table["objective_macro"] = (1 - PRIOR_OOD) * table["cl5_macro"] + PRIOR_OOD * (table["loso_macro"] + table["loco_macro"]) / 2
    table["objective_pooled"] = (1 - PRIOR_OOD) * table["cl5_pooled"] + PRIOR_OOD * (table["loso_pooled"] + table["loco_pooled"]) / 2
    table["regret"] = np.maximum(table["objective_macro"] / table["objective_macro"].min(),
                                 table["objective_pooled"] / table["objective_pooled"].min())
    table = table.sort_values("regret")
    table.to_csv(OUT / "classifier_select.csv", index=False)
    pd.set_option("display.width", 200)
    print(table[["classifier", "cl5_calibrated_log_loss", "objective_macro", "objective_pooled", "regret"]].round(6).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
