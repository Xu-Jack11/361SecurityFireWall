"""Ablation: the same classifier without leakage control (raw text + hour/weekday/month tokens).

Scores it on train-internal unique-document 5-fold, where the leak holds and
so looks harmless or helpful, and on valid, where it does not.

python -m soc_hybrid.ablation_leak [--classifier tfidf_word_lr --weights rows]
  → artifacts/hybrid/ablation_leak.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import CACHE, LABELS, OUT, TRAIN_PATH, VALID_PATH, valid_answers
from .decision import benign_first, cost_matrix, min_cost
from .holdout import cap_cells
from .metrics import evaluate, summary_row
from .models import make_classifier, training_weights
from .splits import unique_doc_folds
from .text import raw_documents, source_keys

MISS_WEIGHTS = (1, 2, 3, 5, 10)
COLUMNS = ["event_id", "timestamp", "pipeline", "vendor_name", "product_name", "src_ip", "dst_ip", "src_port",
           "src_host", "dst_host", "username", "message_sanitized"]


def raw_events(split: str) -> pd.DataFrame:
    path = CACHE / f"{split}_raw_events.parquet"
    if not path.exists():
        frame = pd.read_parquet(TRAIN_PATH if split == "train" else VALID_PATH,
                                columns=COLUMNS + (["label_binary"] if split == "train" else []))
        table = pd.DataFrame({"event_id": frame["event_id"].astype(str), "source": source_keys(frame),
                              "doc": raw_documents(frame)})
        if split == "train":
            table["label"] = frame["label_binary"].astype(str)
        table.to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", default="tfidf_word_lr")
    parser.add_argument("--weights", default="rows")
    parser.add_argument("--cap", type=int, default=20_000)
    args = parser.parse_args()

    tr = raw_events("train")
    unique = tr.groupby(["doc", "label"], sort=False).agg(source=("source", "first"), rows=("event_id", "size")).reset_index()
    report = {"train_unique_raw_docs": int(len(unique))}

    # Train-internal: unique-document 5-fold, as for the leakage-controlled classifier.
    folds = unique_doc_folds(unique)
    proba = np.zeros((len(unique), len(LABELS)))
    for fold in range(5):
        test = folds == fold
        train = cap_cells(unique[~test], args.cap)
        model = make_classifier(args.classifier).fit(train["doc"].tolist(), train["label"].to_numpy(),
                                                     training_weights(train, args.weights))
        proba[test] = model.predict_proba(unique.loc[test, "doc"].tolist())
    for w in (1, 10):
        pred = benign_first(proba, w)
        report[f"ud5_w{w}_doc"] = summary_row(f"ud5 w{w}", evaluate(unique["label"], pred))
        report[f"ud5_w{w}_row"] = summary_row(f"ud5 w{w}", evaluate(unique["label"], pred, unique["rows"]))
    for m in MISS_WEIGHTS:  # corrected rule: minimum expected cost at threat-miss weight m
        pred = min_cost(proba, cost_matrix(m))
        report[f"ud5_m{m}_doc"] = summary_row(f"ud5 m{m}", evaluate(unique["label"], pred))

    # valid: fit on all of train, predict every distinct valid document.
    train = cap_cells(unique, args.cap)
    model = make_classifier(args.classifier).fit(train["doc"].tolist(), train["label"].to_numpy(),
                                                 training_weights(train, args.weights))
    va = raw_events("valid")
    codes, distinct = pd.factorize(va["doc"])
    valid_proba = model.predict_proba(list(distinct))[codes]
    truth = valid_answers().reindex(va["event_id"]).to_numpy()
    for w in (1, 10):
        result = evaluate(truth, benign_first(valid_proba, w))
        report[f"valid_w{w}"] = summary_row(f"valid w{w}", result)
        report[f"valid_w{w}_confusion"] = np.rint(result["confusion"]).astype(int).tolist()
    for m in MISS_WEIGHTS:
        result = evaluate(truth, min_cost(valid_proba, cost_matrix(m)))
        report[f"valid_m{m}"] = summary_row(f"valid m{m}", result)
        report[f"valid_m{m}_confusion"] = np.rint(result["confusion"]).astype(int).tolist()
    (OUT / "ablation_leak.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))



def counterfactual(classifier: str = "tfidf_word_lr", weights: str = "rows", cap: int = 20_000,
                   miss_weight: float | None = None) -> dict:
    """Refit the raw-time classifier on all of train, then move every valid timestamp to one instant.

    A model that reads only the event's content gives the same labels; the share of changed labels measures
    how much the raw model leans on time. No valid label is used.
    """

    tr = raw_events("train")
    unique = tr.groupby(["doc", "label"], sort=False).agg(source=("source", "first"), rows=("event_id", "size")).reset_index()
    train = cap_cells(unique, cap)
    model = make_classifier(classifier).fit(train["doc"].tolist(), train["label"].to_numpy(), training_weights(train, weights))
    frame = pd.read_parquet(VALID_PATH, columns=COLUMNS)
    result = {}
    for name, stamp in (("original", None), ("all at 2024-07-26 11:30 UTC", 1721993400.0)):
        if stamp is not None:
            frame = frame.assign(timestamp=stamp)
        docs = raw_documents(frame)
        codes, distinct = pd.factorize(docs)
        proba = model.predict_proba(list(distinct))[codes]
        result[name] = benign_first(proba, 1.0) if miss_weight is None else min_cost(proba, cost_matrix(miss_weight))
    changed = result["original"] != result["all at 2024-07-26 11:30 UTC"]
    out = {"rows": int(len(changed)), "labels_changed": int(changed.sum()),
           "malicious_before": int((result["original"] == "malicious").sum()),
           "malicious_after": int((result["all at 2024-07-26 11:30 UTC"] == "malicious").sum())}
    suffix = "" if miss_weight is None else f"_m{miss_weight:g}"
    (OUT / f"ablation_leak_counterfactual{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
