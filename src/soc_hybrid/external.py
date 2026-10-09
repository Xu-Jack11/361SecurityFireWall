"""Test the final system (cost2) on random samples of the external WitFoo dumps.

Each external group is sampled uniformly at random (--rows rows, fixed seed). The sampled events go through
exactly the final pipeline: the classifier and novelty models fitted on train, the same triggers with no cap,
the same LLM prompt and the same fusion. Labels are read only to score.

The groups differ in kind (see scripts/eval_external.py and docs/iteration_log.md I3, I10, I14):
  v4           benign/suspicious rows are the train rows verbatim, malicious = train + 14,052 more
  latest       the same events re-sanitized (new pseudonym namespace) and partly relabeled
  v2_live      live capture 07-26..08-01 (valid's period), incident leads labeled malicious in place
  v2_incident  historical incident leads, all malicious
so every group is also scored on two novel subsets: rows whose raw message never occurs in train.parquet
(I14's definition) and rows whose classifier document never occurs in train (stricter: the classifier view
folds pseudonyms and digits, so a re-sanitized train event is not novel to it).

python -m soc_hybrid.external [--rows 10000]  → artifacts/hybrid/external/{summary.json, <group>_rows.parquet}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, OUT, TRAIN_PATH, events, unique_documents
from .decision import cost_confidence, cost_matrix, min_cost
from .llm import LLMRunner, requests_for
from .metrics import evaluate, summary_row
from .pipeline import cost_fusion, fit_classifier, fit_novelty, model_key, route, triggers
from .text import classifier_documents, llm_messages, source_keys

EXTERNAL = Path("data/external")
GROUPS = {
    "v4": EXTERNAL / "witfoo-precinct6-signals-v4.parquet",
    "latest": EXTERNAL / "witfoo-precinct6-signals-latest.parquet",
    "v2_live": EXTERNAL / "precinct6-v2.1.0/signals.parquet",
    "v2_incident": EXTERNAL / "precinct6-v2.1.0/incident_signals.parquet",
}
COLUMNS = ["pipeline", "vendor_name", "product_name", "src_ip", "dst_ip", "src_port", "src_host", "dst_host",
           "username", "message_sanitized", "label_binary"]
KEYS = ["doc", "pipeline", "vendor_name", "product_name", "llm_message"]
RESULT_DIR = OUT / "external"
# The final configuration (cost2 in docs/iteration_log.md, I18).
CAP, MISS_WEIGHT, DOUBT, FUSION, FUSION_PARAM = 20_000, 2.0, 0.9, "mix_km", 0.75


def sample(group: str, rows: int, seed: int = 0) -> pd.DataFrame:
    frame = pd.read_parquet(GROUPS[group], columns=COLUMNS)
    frame = frame.sample(n=min(rows, len(frame)), random_state=seed).reset_index(drop=True)
    return pd.DataFrame({
        "event_id": [f"{group}-{i}" for i in range(len(frame))],
        "source": source_keys(frame),
        "doc": classifier_documents(frame),
        "llm_message": llm_messages(frame),
        "pipeline": frame["pipeline"].fillna("").astype(str),
        "vendor_name": frame["vendor_name"].fillna("").astype(str),
        "product_name": frame["product_name"].fillna("").astype(str),
        "message": frame["message_sanitized"],
        "label": frame["label_binary"].astype(str),
    })


def load_models() -> tuple:
    """The final system's classifiers (cached by pipeline.py) and the novelty model."""

    cnn_params = {"device": "cuda"}
    cnn = fit_classifier("textcnn", "rows", CAP, cnn_params, model_key("textcnn", "rows", CAP, cnn_params))
    tfidf = fit_classifier("tfidf_word_lr", "rows", CAP, {}, model_key("tfidf_word_lr", "rows", CAP, {}))
    return cnn, tfidf, fit_novelty(CAP)


def classify(docs: pd.DataFrame, train_cells: set, models: tuple) -> pd.DataFrame:
    """Classifier probabilities, decision, confidence, novelty and triggers for distinct documents."""

    cnn, tfidf, novelty = models
    texts = docs["doc"].tolist()
    proba = (cnn.predict_proba(texts) + tfidf.predict_proba(texts)) / 2
    for i, label in enumerate(LABELS):
        docs[f"p_{label}"] = proba[:, i]
    docs["clf"] = min_cost(proba, cost_matrix(MISS_WEIGHT))
    docs["conf"] = cost_confidence(proba, cost_matrix(MISS_WEIGHT))
    docs["maxsim"] = novelty.max_similarity(texts)
    fired = triggers(docs, DOUBT, 0.0, True, True, train_cells)
    for name in fired.columns:
        docs[f"trigger_{name}"] = fired[name].to_numpy()
    docs["routed"] = route(docs, fired, budget=0, priority="doubt")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train = unique_documents(events("train"))
    train_cells = set(zip(train["source"], train["label"]))
    train_docs = set(train["doc"])
    train_messages = set(pd.read_parquet(TRAIN_PATH, columns=["message_sanitized"])["message_sanitized"].dropna())

    models = load_models()
    samples, documents = {}, {}
    for group in GROUPS:
        rows = sample(group, args.rows, args.seed)
        rows["novel_raw"] = ~rows["message"].isin(train_messages)
        rows["novel_doc"] = ~rows["doc"].isin(train_docs)
        docs = rows.groupby(KEYS, sort=False).agg(rows=("event_id", "size"), source=("source", "first")).reset_index()
        samples[group], documents[group] = rows, classify(docs, train_cells, models)
        print(f"{group}: {len(rows)} rows, {len(docs)} distinct documents, {int(docs['routed'].sum())} routed")
    if getattr(models[0], "net", None) is not None:  # free the GPU for vLLM
        import torch

        models[0].net.to("cpu")
        torch.cuda.empty_cache()

    # One LLM run over every group's routed documents (the runner caches replies in llm/external.jsonl).
    runner = LLMRunner("external")
    routed = pd.concat([documents[g][documents[g]["routed"]].assign(group=g) for g in GROUPS], ignore_index=False)
    replies = runner.run(requests_for(routed, "codebook"))
    routed["llm"] = [reply["label"] for reply in replies]
    for label in LABELS:
        routed[f"llm_p_{label}"] = [(reply["probs"] or {}).get(label) if reply.get("probs") else float(reply["label"] == label)
                                    for reply in replies]

    summary = {"rows_per_group": args.rows, "seed": args.seed, "groups": {}}
    for group in GROUPS:
        docs = documents[group]
        part = routed[routed["group"] == group]
        docs["llm"] = None
        for label in LABELS:
            docs[f"llm_p_{label}"] = np.nan
        docs.loc[part.index, "llm"] = part["llm"].to_numpy()
        for label in LABELS:
            docs.loc[part.index, f"llm_p_{label}"] = part[f"llm_p_{label}"].to_numpy()
        proba = docs[[f"p_{label}" for label in LABELS]].to_numpy()
        llm_proba = docs[[f"llm_p_{label}" for label in LABELS]].to_numpy(dtype=float)
        docs["final"] = cost_fusion(docs, proba, llm_proba, FUSION, FUSION_PARAM, MISS_WEIGHT)
        rows = samples[group].merge(docs[KEYS + ["clf", "final", "routed", "conf", "maxsim"]], on=KEYS, how="left")
        rows.drop(columns=["doc", "llm_message", "message"]).to_parquet(RESULT_DIR / f"{group}_rows.parquet", index=False)
        report = {"labels": rows["label"].value_counts().to_dict(), "distinct_documents": int(len(docs)),
                  "routed_documents": int(docs["routed"].sum()), "routed_rows": int(rows["routed"].sum()),
                  "llm_changed_rows": int((rows["final"] != rows["clf"]).sum())}
        for subset, keep in (("all", np.ones(len(rows), dtype=bool)), ("novel_raw", rows["novel_raw"].to_numpy()),
                             ("novel_doc", rows["novel_doc"].to_numpy())):
            if not keep.any():
                continue
            truth = rows.loc[keep, "label"]
            final, clf = evaluate(truth, rows.loc[keep, "final"]), evaluate(truth, rows.loc[keep, "clf"])
            report[subset] = {"rows": int(keep.sum()), "labels": truth.value_counts().to_dict(),
                              "system": summary_row("system", final), "classifier_only": summary_row("classifier only", clf),
                              "system_confusion": np.rint(final["confusion"]).astype(int).tolist(),
                              "system_per_class": {k: {m: round(float(v[m]), 4) for m in ("precision", "recall")}
                                                   for k, v in final["per_class"].items()}}
        summary["groups"][group] = report
        print(f"\n== {group}: routed {report['routed_documents']} docs / {report['routed_rows']} rows, "
              f"LLM changed {report['llm_changed_rows']} rows")
        for subset in ("all", "novel_raw", "novel_doc"):
            if subset in report:
                s, c = report[subset]["system"], report[subset]["classifier_only"]
                print(f"  {subset:9s} rows {report[subset]['rows']:6d} labels {report[subset]['labels']}")
                print(f"     system:     misses {s['threat_misses']:5d} FA {s['benign_misses']:5d} swaps {s['swaps']:5d} "
                      f"macro-F1 {s['macro_f1']:.4f} threat recall {s['threat_recall']:.4f} cost m2/5/10 "
                      f"{s['cost_m2']:.0f}/{s['cost_m5']:.0f}/{s['cost_m10']:.0f}")
                print(f"     classifier: misses {c['threat_misses']:5d} FA {c['benign_misses']:5d} swaps {c['swaps']:5d} "
                      f"macro-F1 {c['macro_f1']:.4f} threat recall {c['threat_recall']:.4f} cost m2/5/10 "
                      f"{c['cost_m2']:.0f}/{c['cost_m5']:.0f}/{c['cost_m10']:.0f}")
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
