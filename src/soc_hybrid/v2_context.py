"""Would grouping v2's records by source first, then judging the group, separate malicious from suspicious? (I22)

Three checks on v2 live's threat rows, reading v2's own labels (never valid):

  incidents  what the members of one Precinct incident share (source IP, destination IP, product)
  groups     how the labels mix inside groups of one source: the source IP over the whole capture, per hour, per
             10 minutes, source + destination per hour, destination per hour. The in-sample majority label per group
             bounds any method that gives a group one verdict, an LLM reading the whole group included
  context    record-level models fitted on 07-26 and scored on the later days: the record's content (TF-IDF over
             what the LLM reads), label-free statistics of its (source IP, hour) and (destination IP, hour) groups
             computed over all records, and both (gradient boosting on the context plus a cross-fitted content score)

python -m soc_hybrid.v2_context  → artifacts/hybrid/v2_context/summary.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

from .data import OUT
from .external import COLUMNS, GROUPS
from .text import llm_messages

RESULTS = OUT / "v2_context"
PRODUCTS = ("ASA Firewall", "AWS VPC Security")
GROUPINGS = {"source IP": ["src_ip"], "source IP x 1 h": ["src_ip", "hour"], "source IP x 10 min": ["src_ip", "minute10"],
             "source + destination IP x 1 h": ["src_ip", "dst_ip", "hour"], "destination IP x 1 h": ["dst_ip", "hour"]}


def _load() -> pd.DataFrame:
    live = pd.read_parquet(GROUPS["v2_live"], columns=COLUMNS + ["timestamp", "dst_port", "incident_ids"])
    live = live.astype({c: object for c in COLUMNS})
    time = pd.to_datetime(live["timestamp"], unit="s", utc=True)
    message = live["message_sanitized"].fillna("")
    live = live.assign(time=time, hour=time.dt.floor("1h"), minute10=time.dt.floor("10min"), day=time.dt.strftime("%m-%d"),
                       deny=message.str.contains(r"\bDeny\b|REJECT|,drop,", regex=True).astype(int),
                       inbound=message.str.contains(r"Deny \w+ src outside:", regex=True).astype(int))
    by_source = live.groupby(["src_ip", "hour"], dropna=False)
    by_target = live.groupby(["dst_ip", "hour"], dropna=False)
    # Label-free context over every record, benign included: what a SOC sees around an alert.
    context = {"source_records": by_source["time"].transform("size"), "source_denies": by_source["deny"].transform("sum"),
               "source_inbound": by_source["inbound"].transform("sum"), "source_targets": by_source["dst_ip"].transform("nunique"),
               "source_ports": by_source["dst_port"].transform("nunique"),
               "source_products": by_source["product_name"].transform("nunique"),
               "source_minutes": by_source["time"].transform(lambda t: (t.max() - t.min()).total_seconds() / 60),
               "target_records": by_target["time"].transform("size"), "target_sources": by_target["src_ip"].transform("nunique")}
    live = live.assign(**{f"ctx_{name}": values for name, values in context.items()})
    threats = live[live["label_binary"] != "benign"].reset_index(drop=True)
    return threats.assign(y=threats["label_binary"].eq("malicious").astype(int))


def incidents(threats: pd.DataFrame) -> dict:
    members = threats.loc[threats["y"] == 1].assign(incident=lambda d: d["incident_ids"].map(json.loads)).explode("incident")
    shared = members.groupby("incident").agg(rows=("src_ip", "size"), sources=("src_ip", "nunique"),
                                             targets=("dst_ip", "nunique"), products=("product_name", "nunique"))
    multi = shared[shared["rows"] >= 2]
    return {"incidents": int(len(shared)), "with_2_or_more_live_rows": int(len(multi)),
            "one_source_ip": round(float((multi["sources"] == 1).mean()), 3),
            "one_destination_ip": round(float((multi["targets"] == 1).mean()), 3),
            "one_product": round(float((multi["products"] == 1).mean()), 3)}


def groups(threats: pd.DataFrame) -> dict:
    malicious = threats["y"].to_numpy() == 1
    report = {}
    for name, keys in GROUPINGS.items():
        grouped = threats.groupby(keys, dropna=False)["y"]
        share = grouped.transform("mean").to_numpy()
        majority = share > 0.5
        entry = {"groups": int(grouped.ngroups),
                 "malicious_rows_in_all_malicious_groups": round(float((malicious & (share == 1)).sum() / malicious.sum()), 3),
                 "suspicious_rows_in_groups_with_malicious": round(float((~malicious & (share > 0)).sum() / (~malicious).sum()), 3),
                 "majority_oracle_malicious_recall": round(float((majority & malicious).sum() / malicious.sum()), 3),
                 "majority_oracle_malicious_precision": round(float((majority & malicious).sum() / max(majority.sum(), 1)), 3)}
        for product in PRODUCTS:
            keep = threats["product_name"].eq(product).to_numpy()
            entry[f"majority_oracle_malicious_recall_{product}"] = round(
                float((majority & malicious & keep).sum() / (malicious & keep).sum()), 3)
        report[name] = entry
    return report


def _score(y: np.ndarray, p: np.ndarray, product: np.ndarray) -> dict:
    precision, recall, _ = precision_recall_curve(y, p)
    entry = {"auc": round(float(roc_auc_score(y, p)), 3), "average_precision": round(float(average_precision_score(y, p)), 3),
             "best_f1": round(float((2 * precision * recall / np.maximum(precision + recall, 1e-9)).max()), 3),
             "recall_at_precision_50": round(float(recall[precision >= 0.5].max()), 3)}
    for name in PRODUCTS:
        keep = product == name
        entry[f"auc_{name}"] = round(float(roc_auc_score(y[keep], p[keep])), 3)
        entry[f"average_precision_{name}"] = round(float(average_precision_score(y[keep], p[keep])), 3)
    return entry


def context(threats: pd.DataFrame) -> dict:
    train = threats["day"].eq("07-26").to_numpy()
    y = threats["y"].to_numpy()
    text = threats["vendor_name"].fillna("") + "|" + threats["product_name"].fillna("") + "|" + llm_messages(threats)
    features = np.log1p(threats[[c for c in threats.columns if c.startswith("ctx_")]].to_numpy(dtype=float))
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s|]+", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
    x_train, x_test = vectorizer.fit_transform(text[train]), vectorizer.transform(text[~train])
    content = LogisticRegression(C=4.0, max_iter=2000)
    p_content = content.fit(x_train, y[train]).predict_proba(x_test)[:, 1]
    # Cross-fitted on 07-26 so the stacked model does not learn to trust an overfitted score.
    p_content_train = cross_val_predict(LogisticRegression(C=4.0, max_iter=2000), x_train, y[train], cv=5,
                                        method="predict_proba")[:, 1]
    boost = dict(max_iter=300, learning_rate=0.05, random_state=0)
    p_context = HistGradientBoostingClassifier(**boost).fit(features[train], y[train]).predict_proba(features[~train])[:, 1]
    p_both = (HistGradientBoostingClassifier(**boost).fit(np.column_stack([features[train], p_content_train]), y[train])
              .predict_proba(np.column_stack([features[~train], p_content]))[:, 1])
    product = threats.loc[~train, "product_name"].to_numpy()
    return {"test_rows": int((~train).sum()), "test_malicious": int(y[~train].sum()),
            "content": _score(y[~train], p_content, product), "same-source context": _score(y[~train], p_context, product),
            "content + context": _score(y[~train], p_both, product)}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    threats = _load()
    summary = {"incidents": incidents(threats), "groups": groups(threats), "context": context(threats)}
    print("incidents:", summary["incidents"])
    for name, entry in summary["groups"].items():
        print(f"{name:30s} {entry}")
    for name, entry in summary["context"].items():
        print(f"{name:20s} {entry}")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
