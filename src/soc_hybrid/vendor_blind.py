"""Refit the final classifier without the vendor and product fields (I21): what it learns instead, how it transfers.

I20 found that the final classifier types threats by the vendor field: in train every malicious row lacks vendor and
product and every suspicious row has them. Here the same classifier (TextCNN + TF-IDF mean, rows weighting,
20,000-document cap, minimum cost at m = 2) is refitted on documents without the fvendor_/fproduct_ tokens, and both
versions are compared classifier-only (no routing, no LLM; the LLM never picks the alert type anyway):

  valid      the blind classifier's valid predictions are written as run ``vendor_blind`` for soc_hybrid.evaluate,
             the step that reads valid labels; the original is cost2's classifier-only column
  external   the I19 samples (10,000 rows per group, seed 0) and every v2 threat row
  learned    how much of train's suspicious/malicious split survives without the field, and the blind TF-IDF
             model's strongest malicious-vs-suspicious n-grams

python -m soc_hybrid.vendor_blind  → artifacts/hybrid/vendor_blind/summary.json, runs/vendor_blind/docs.parquet
python -m soc_hybrid.evaluate --runs vendor_blind,cost2
"""

from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import LABELS, OUT, events, unique_documents
from .decision import cost_matrix, min_cost
from .external import COLUMNS, GROUPS, MISS_WEIGHT, load_models, sample
from .holdout import cap_cells
from .metrics import evaluate, summary_row
from .models import make_classifier, training_weights
from .pipeline import MODELS, RUNS, valid_documents
from .text import classifier_documents

RESULTS = OUT / "vendor_blind"
RUN = "vendor_blind"
CAP = 20_000
CLASSIFIERS = (("textcnn", {"device": "cuda"}), ("tfidf_word_lr", {}))
THREAT_PRODUCTS = ("ASA Firewall", "AWS VPC Security")
_VENDOR = re.compile(r"\s*\bf(?:vendor|product)_\S+")


def _auc(y, score) -> float | None:
    y = np.asarray(y)
    return round(float(roc_auc_score(y, score)), 3) if len(set(y)) == 2 else None


def blind(docs) -> list[str]:
    """Classifier documents without the vendor and product tokens (the message after ' | ' is untouched)."""

    codes, uniques = pd.factorize(pd.Series(docs, dtype=object))
    stripped = []
    for doc in uniques:
        head, sep, tail = doc.partition(" | ")
        stripped.append(_VENDOR.sub("", head).strip() + sep + tail)
    return np.asarray(stripped, dtype=object)[codes].tolist()


def training_set() -> pd.DataFrame:
    """The final classifier's training documents (capped per cell as before), blinded and regrouped."""

    train = cap_cells(unique_documents(events("train")), CAP)
    train = train.assign(doc=blind(train["doc"]))
    return train.groupby(["doc", "label"], sort=False).agg(rows=("rows", "sum"), source=("source", "first")).reset_index()


def fit_blind() -> tuple:
    train = None
    models = []
    for classifier, params in CLASSIFIERS:
        path = MODELS / f"{classifier}_rows_cap{CAP}_novendor.joblib"
        if path.exists():
            models.append(joblib.load(path))
            continue
        train = training_set() if train is None else train
        model = make_classifier(classifier, **params).fit(train["doc"].tolist(), train["label"].to_numpy(),
                                                           training_weights(train, "rows"))
        joblib.dump(model, path)
        models.append(model)
    return tuple(models)


def predict(models: tuple, docs: list[str]) -> np.ndarray:
    codes, uniques = pd.factorize(pd.Series(docs, dtype=object))
    texts = list(uniques)
    return ((models[0].predict_proba(texts) + models[1].predict_proba(texts)) / 2)[codes]


def write_valid_run(models: tuple) -> dict:
    """Blind classifier-only predictions for valid in valid_documents() order, as evaluate.row_frame expects."""

    docs = valid_documents()
    proba = predict(models, blind(docs["doc"]))
    decided = min_cost(proba, cost_matrix(MISS_WEIGHT))
    frame = pd.DataFrame({"source": docs["source"].to_numpy(), "rows": docs["rows"].to_numpy(), "clf": decided,
                          "final": decided, "routed": False,
                          **{f"p_{label}": proba[:, i] for i, label in enumerate(LABELS)}})
    (RUNS / RUN).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RUNS / RUN / "docs.parquet", index=False)
    config = {"classifier": "textcnn + tfidf_word_lr (rows, cap 20,000)", "vendor_fields": "dropped",
              "miss_weight": MISS_WEIGHT, "llm": "none"}
    (RUNS / RUN / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"documents": int(len(frame)), "predicted_rows": {label: int(frame.loc[frame["clf"] == label, "rows"].sum())
                                                             for label in LABELS}}


def external(original: tuple, blinded: tuple) -> dict:
    """Both classifiers on the I19 samples, plus every v2 threat row."""

    versions = {"original": (original, lambda docs: list(docs)), "no vendor": (blinded, blind)}
    report = {"samples": {}, "v2_threat_rows": {}}
    for group in GROUPS:
        rows = sample(group, 10_000, 0)
        report["samples"][group] = {}
        for name, (models, view) in versions.items():
            decided = min_cost(predict(models, view(rows["doc"])), cost_matrix(MISS_WEIGHT))
            result = evaluate(rows["label"], decided)
            entry = summary_row(name, result)
            entry["malicious_recall"] = round(result["per_class"]["malicious"]["recall"], 4)
            entry["malicious_precision"] = round(result["per_class"]["malicious"]["precision"], 4)
            entry["confusion"] = np.rint(result["confusion"]).astype(int).tolist()
            report["samples"][group][name] = entry

    live = pd.read_parquet(GROUPS["v2_live"], columns=COLUMNS)
    live = live[live["label_binary"] != "benign"].reset_index(drop=True).astype({c: object for c in COLUMNS})
    incident = pd.read_parquet(GROUPS["v2_incident"], columns=COLUMNS).astype({c: object for c in COLUMNS})
    malicious = live["label_binary"].eq("malicious").to_numpy()
    zone = live["message_sanitized"].str.extract(r"Deny \w+ src ([\w-]+):")[0].str.lower().to_numpy()
    live_docs, incident_docs = classifier_documents(live), classifier_documents(incident)
    for name, (models, view) in versions.items():
        proba = predict(models, view(live_docs))
        decided = min_cost(proba, cost_matrix(MISS_WEIGHT))
        entry = {f"live_true_{truth}": {label: round(float((decided[keep] == label).mean()), 4) for label in LABELS}
                 for truth, keep in (("malicious", malicious), ("suspicious", ~malicious))}
        entry["live_auc_p_malicious"] = _auc(malicious, proba[:, 2])
        for product in THREAT_PRODUCTS:
            keep = live["product_name"].eq(product).to_numpy()
            entry[f"live_auc_{product}"] = _auc(malicious[keep], proba[keep, 2])
        asa = live["product_name"].eq("ASA Firewall").to_numpy()
        entry["live_asa_called_malicious_by_zone"] = {
            z: {truth: round(float((decided[asa & (zone == z) & keep] == "malicious").mean()), 4)
                for truth, keep in (("malicious", malicious), ("suspicious", ~malicious))}
            for z in ("outside", "dmz-2")}
        decided = min_cost(predict(models, view(incident_docs)), cost_matrix(MISS_WEIGHT))
        entry["incident_called"] = {label: round(float((decided == label).mean()), 4) for label in LABELS}
        entry["incident_malicious_by_product"] = {
            product: round(float((decided[incident["product_name"].eq(product).to_numpy()] == "malicious").mean()), 4)
            for product in ("ASA Firewall", "Meraki", "PAN NGFW", "AWS VPC Security")}
        report["v2_threat_rows"][name] = entry
    return report


def learned(blinded: tuple) -> dict:
    """What separates train's threats once the vendor field is gone."""

    train = unique_documents(events("train"))
    threats = train[train["label"] != "benign"].assign(doc=blind(train.loc[train["label"] != "benign", "doc"]))
    labels_per_doc = threats.groupby("doc")["label"].nunique()
    shared = threats["doc"].map(labels_per_doc > 1)
    malicious = threats["label"] == "malicious"
    report = {
        "train_malicious_rows_sharing_blind_doc_with_suspicious":
            round(float(threats.loc[malicious & shared, "rows"].sum() / threats.loc[malicious, "rows"].sum()), 4),
        "train_suspicious_rows_sharing_blind_doc_with_malicious":
            round(float(threats.loc[~malicious & shared, "rows"].sum() / threats.loc[~malicious, "rows"].sum()), 4),
    }
    tfidf = blinded[1]
    names = tfidf.vectorizer.get_feature_names_out()
    coef = np.zeros((len(LABELS), len(names)))
    coef[tfidf.model.classes_] = tfidf.model.coef_
    margin = coef[2] - coef[1]  # malicious minus suspicious
    order = np.argsort(margin)
    report["top_malicious_ngrams"] = [[names[i], round(float(margin[i]), 2)] for i in order[::-1][:15]]
    report["top_suspicious_ngrams"] = [[names[i], round(float(margin[i]), 2)] for i in order[:15]]
    return report


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    original = load_models()[:2]
    blinded = fit_blind()
    summary = {"valid_run": write_valid_run(blinded)}
    print("valid run written:", summary["valid_run"])
    summary["external"] = external(original, blinded)
    for group, versions in summary["external"]["samples"].items():
        for name, entry in versions.items():
            print(f"{group:12s} {name:10s} misses {entry['threat_misses']:5d} FA {entry['benign_misses']:5d} "
                  f"swaps {entry['swaps']:5d} macro-F1 {entry['macro_f1']:.4f} mal R/P {entry['malicious_recall']:.3f}/"
                  f"{entry['malicious_precision']:.3f} cost m2/5/10 {entry['cost_m2']:.0f}/{entry['cost_m5']:.0f}/{entry['cost_m10']:.0f}")
    for name, entry in summary["external"]["v2_threat_rows"].items():
        print(f"v2 threat rows, {name}: {entry}")
    summary["learned"] = learned(blinded)
    print("learned:", json.dumps(summary["learned"], ensure_ascii=False))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
