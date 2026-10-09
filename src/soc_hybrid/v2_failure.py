"""Why the final system types every Precinct6 v2 malicious row as suspicious (I20).

The external test (I19) found no threat misses in any group, but v2's malicious rows all came out suspicious.
This script measures why. It reads the dumps' own labeling columns (lifecycle_stage, matched_rules,
incident_ids), which the competition data do not carry, and never touches valid:

  labels       how each dump's labels line up with Precinct's correlation output and with empty vendor fields
  same_events  v4's malicious rows (the competition's malicious class) against v2's incident_signals,
               paired on (timestamp, src_port, dst_port)
  swap         the final classifier on those pairs with the message format and the vendor fields swapped (2 x 2),
               and on every v2 threat row with and without its vendor fields
  separable    whether a single v2 record separates malicious from suspicious: shared views, the in-sample best
               any function of a view can do, TF-IDF models fitted on v2's own labels, past-only context counts
  system       what the routing triggers saw in the I19 run (artifacts/hybrid/external/*_rows.parquet)
  leads        why identical v2 records carry both labels: the dataset card calls a row malicious only if Precinct
               attached that artifact to an incident as a lead, and Precinct attaches few records of a repeated activity

python -m soc_hybrid.v2_failure [--parts leads,...]  → artifacts/hybrid/v2_failure/summary.json
(--parts reruns only the named parts and keeps the others' saved results)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .data import LABELS, OUT, TRAIN_PATH
from .decision import cost_matrix, min_cost
from .external import COLUMNS, GROUPS, MISS_WEIGHT, RESULT_DIR, load_models
from .text import classifier_documents, llm_messages

RESULTS = OUT / "v2_failure"
PAIR_KEY = ["ts", "src_port", "dst_port"]
PRODUCTS = ("ASA Firewall", "Meraki", "PAN NGFW", "AWS VPC Security")
THREAT_PRODUCTS = ("ASA Firewall", "AWS VPC Security")  # 99.7% of v2's live threat rows
VIEWS = ("doc", "llm", "raw")


def _share(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return round(float(mask.mean()), 4) if len(mask) else float("nan")


def _nonempty(values: pd.Series) -> np.ndarray:
    """JSON list columns: '[]' (or missing) when empty."""

    return values.fillna("[]").astype(str).str.strip().ne("[]").to_numpy()


def _auc(y, score) -> float | None:
    y = np.asarray(y)
    return round(float(roc_auc_score(y, score)), 3) if len(set(y)) == 2 else None


def labels() -> dict:
    """Per dump and label: rows, share without vendor, with a matched rule, linked to an incident, lifecycle stages."""

    report = {}
    train = pd.read_parquet(TRAIN_PATH, columns=["label_binary", "vendor_name"])
    report["train"] = {label: {"rows": int(len(part)), "no_vendor": _share(part["vendor_name"].fillna("").str.strip().eq(""))}
                       for label, part in train.groupby("label_binary")}
    for group, path in GROUPS.items():
        names = pq.ParquetFile(path).schema_arrow.names
        frame = pd.read_parquet(path, columns=["label_binary", "vendor_name", "lifecycle_stage", "matched_rules"]
                                + (["incident_ids"] if "incident_ids" in names else []))
        frame["no_vendor"] = frame["vendor_name"].fillna("").astype(str).str.strip().eq("")
        frame["rule_hit"] = _nonempty(frame["matched_rules"])
        frame["incident"] = _nonempty(frame["incident_ids"]) if "incident_ids" in frame else False
        report[group] = {label: {"rows": int(len(part)), "no_vendor": _share(part["no_vendor"]),
                                 "rule_hit": _share(part["rule_hit"]),
                                 "incident": _share(part["incident"]) if "incident_ids" in frame else None,
                                 "stages": part["lifecycle_stage"].astype(str).value_counts().to_dict()}
                         for label, part in frame.groupby("label_binary")}

    live = pd.read_parquet(GROUPS["v2_live"], columns=["label_binary", "incident_ids", "disposition", "action"])
    incident = pd.read_parquet(GROUPS["v2_incident"], columns=["incident_ids"])
    linked = live.loc[live["label_binary"] == "malicious", ["incident_ids", "disposition"]]
    members = linked["incident_ids"].map(json.loads).explode()
    history = set(incident["incident_ids"].map(json.loads).explode())
    sizes = members.value_counts()
    report["v2_live_incidents"] = {
        "incidents": int(len(sizes)), "rows_per_incident_p50_p90_p99": sizes.quantile([0.5, 0.9, 0.99]).tolist(),
        "also_in_incident_signals": int(len(set(sizes.index) & history)),
        "malicious_disposition": linked["disposition"].value_counts().to_dict(),
        "block_action_by_label": live.loc[live["action"].eq("block"), "label_binary"].value_counts().to_dict(),
    }
    return report


def same_events() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """v4's malicious rows against v2's incident_signals: timestamp coverage and one pair per event.

    v2 rebuilt the pseudonym registry, so IPs differ; (second, src_port, dst_port) is unique enough on the v2
    side, while v4 repeats most events 2-8 times (the first copy is kept).
    """

    columns = COLUMNS + ["timestamp", "dst_port"]
    v4 = pd.read_parquet(GROUPS["v4"], columns=columns)
    v4 = v4[v4["label_binary"] == "malicious"].copy()
    incident = pd.read_parquet(GROUPS["v2_incident"], columns=columns)
    for frame in (v4, incident):
        frame["ts"] = frame["timestamp"].astype(float).round().astype("int64")
    counts = incident.groupby(PAIR_KEY).size()
    first = v4.drop_duplicates(PAIR_KEY).set_index(PAIR_KEY)
    keys = counts[counts == 1].index.intersection(first.index).sort_values()
    old = first.loc[keys].reset_index()
    new = incident.set_index(PAIR_KEY).loc[keys].reset_index()
    example = int(np.flatnonzero(new["product_name"].eq("ASA Firewall").to_numpy())[0])
    report = {
        "v4_malicious_rows": int(len(v4)), "v2_incident_rows": int(len(incident)),
        "v4_malicious_timestamps_in_v2_incident": _share(v4["ts"].isin(set(incident["ts"]))),
        "v4_malicious_with_vendor": _share(v4["vendor_name"].fillna("").str.strip().ne("")),
        "pairs": int(len(keys)), "pair_products": new["product_name"].value_counts().to_dict(),
        "example": {"v4": {"vendor": old.at[example, "vendor_name"], "product": old.at[example, "product_name"],
                           "message": str(old.at[example, "message_sanitized"]).strip()},
                    "v2": {"vendor": new.at[example, "vendor_name"], "product": new.at[example, "product_name"],
                           "message": str(new.at[example, "message_sanitized"]).strip()}},
    }
    return report, old, new


def decide(frame: pd.DataFrame, models: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The final classifier on raw rows: TextCNN + TF-IDF mean, minimum cost at m = 2 (as in external.classify)."""

    frame = frame.reset_index(drop=True).astype({c: object for c in COLUMNS if c in frame.columns})
    codes, texts = pd.factorize(classifier_documents(frame))
    cnn, tfidf, _ = models
    proba = ((cnn.predict_proba(list(texts)) + tfidf.predict_proba(list(texts))) / 2)[codes]
    return min_cost(proba, cost_matrix(MISS_WEIGHT)), proba


def _vendor(frame: pd.DataFrame, source: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rows with the vendor and product of ``source`` (blank when None)."""

    frame = frame.copy()
    for column in ("vendor_name", "product_name"):
        frame[column] = "" if source is None else source[column].to_numpy()
    return frame


def swap(models: tuple, old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Message format x vendor fields on the paired events, and v2's threat rows with their vendor fields blanked."""

    cells = {"v4 message, no vendor (v4 as is)": old, "v4 message + v2 vendor": _vendor(old, new),
             "v2 message, vendor blanked": _vendor(new), "v2 message + vendor (v2 as is)": new}
    product = new["product_name"].to_numpy()
    pairs = {}
    for name, frame in cells.items():
        decided, _ = decide(frame, models)
        pairs[name] = {label: _share(decided == label) for label in LABELS}
        pairs[name]["malicious_by_product"] = {p: _share(decided[product == p] == "malicious") for p in PRODUCTS
                                               if (product == p).any()}

    live = pd.read_parquet(GROUPS["v2_live"], columns=COLUMNS)
    live = live[live["label_binary"] != "benign"].reset_index(drop=True)
    malicious = live["label_binary"].eq("malicious").to_numpy()
    incident = pd.read_parquet(GROUPS["v2_incident"], columns=COLUMNS)
    threat_rows = {}
    for name, frame, blank in (("live as is", live, False), ("live vendor blanked", live, True),
                               ("incident as is", incident, False), ("incident vendor blanked", incident, True)):
        decided, proba = decide(_vendor(frame) if blank else frame, models)
        if name.startswith("live"):
            entry = {f"true_{truth}": {label: _share(decided[keep] == label) for label in LABELS}
                     for truth, keep in (("malicious", malicious), ("suspicious", ~malicious))}
            entry["auc_p_malicious"] = _auc(malicious, proba[:, 2])
            for p in THREAT_PRODUCTS:
                keep = live["product_name"].eq(p).to_numpy()
                entry[f"auc_p_malicious_{p}"] = _auc(malicious[keep], proba[keep, 2])
        else:
            entry = {label: _share(decided == label) for label in LABELS}
        threat_rows[name] = entry
    return {"pairs": pairs, "threat_rows": threat_rows, "live_threat_rows": int(len(live)),
            "live_malicious_rows": int(malicious.sum()), "incident_rows": int(len(incident))}


def _live_threats() -> pd.DataFrame:
    live = pd.read_parquet(GROUPS["v2_live"], columns=COLUMNS + ["timestamp"])
    live = live.sort_values("timestamp", kind="stable").reset_index(drop=True)
    # Past-only context: earlier records (any label) with the same source, destination or both.
    live["past_pair"] = live.groupby(["src_ip", "dst_ip"], dropna=False).cumcount()
    live["past_src"] = live.groupby("src_ip", dropna=False).cumcount()
    live["past_dst"] = live.groupby("dst_ip", dropna=False).cumcount()
    live = live[live["label_binary"] != "benign"].reset_index(drop=True).astype({c: object for c in COLUMNS})
    live["y"] = live["label_binary"].eq("malicious").astype(int)
    live["doc"] = classifier_documents(live)

    def text(name: str) -> pd.Series:
        return live[name].fillna("").astype(str)

    live["llm"] = text("pipeline") + "|" + text("vendor_name") + "|" + text("product_name") + "|" + llm_messages(live)
    live["raw"] = (text("vendor_name") + "|" + text("product_name") + "|" + text("src_ip") + "|" + text("dst_ip")
                   + "|" + text("message_sanitized"))
    live["day"] = pd.to_datetime(live["timestamp"], unit="s", utc=True).dt.strftime("%m-%d")
    live["zone"] = live["message_sanitized"].str.extract(r"Deny \w+ src ([\w-]+):")[0].str.lower()
    return live


def _fit(live: pd.DataFrame, view: str, train: np.ndarray) -> dict:
    """TF-IDF + logistic regression fitted on v2's own labels, scored on the other rows."""

    vectorizer = TfidfVectorizer(token_pattern=r"[^\s|]+", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 max_features=300_000)
    x_train = vectorizer.fit_transform(live.loc[train, view])
    model = LogisticRegression(C=4.0, max_iter=2000).fit(x_train, live.loc[train, "y"])
    p = model.predict_proba(vectorizer.transform(live.loc[~train, view]))[:, 1]
    y = live.loc[~train, "y"].to_numpy()
    precision, recall, _ = precision_recall_curve(y, p)
    entry = {"test_rows": int(len(y)), "test_malicious": int(y.sum()), "auc": _auc(y, p),
             "average_precision": round(float(average_precision_score(y, p)), 3),
             "best_f1": round(float((2 * precision * recall / np.maximum(precision + recall, 1e-9)).max()), 3),
             "recall_at_precision_50": round(float(recall[precision >= 0.5].max()), 3),
             "called_malicious": int((p > 0.5).sum()), "right_at_0.5": int(((p > 0.5) & (y == 1)).sum())}
    for product in THREAT_PRODUCTS:
        keep = (live.loc[~train, "product_name"] == product).to_numpy()
        entry[f"auc_{product}"] = _auc(y[keep], p[keep])
    return entry


def separable() -> dict:
    live = _live_threats()
    report = {"views": {}, "in_domain": {}, "context": {}}
    for view in VIEWS:
        counts = live.groupby(view)["y"].agg(["sum", "size"])
        malicious, suspicious = counts["sum"], counts["size"] - counts["sum"]
        shared = live[view].map((malicious > 0) & (suspicious > 0))
        report["views"][view] = {"distinct": int(len(counts)),
                                 "malicious_rows_sharing_view_with_suspicious": _share(shared[live["y"] == 1]),
                                 "oracle_malicious_recall": round(float(malicious[malicious > suspicious].sum() / live["y"].sum()), 3)}
    asa = live[live["product_name"] == "ASA Firewall"]
    zones = pd.crosstab(asa["zone"].fillna("other"), asa["label_binary"])
    report["asa_source_zone"] = {zone: {"malicious": int(row.get("malicious", 0)), "suspicious": int(row.get("suspicious", 0)),
                                        "malicious_share": round(float(row.get("malicious", 0) / row.sum()), 3)}
                                 for zone, row in zones.iterrows() if row.sum() >= 100}
    splits = {"day (07-26 -> later)": live["day"].eq("07-26").to_numpy(),
              "random 80/20": np.random.default_rng(0).random(len(live)) < 0.8}
    for split, train in splits.items():
        report["in_domain"][split] = {view: _fit(live, view, train) for view in VIEWS}
    inbound = (live["product_name"] == "ASA Firewall") & (live["zone"] == "outside")
    for name, keep in (("ASA inbound (src zone outside)", inbound), ("AWS VPC Security", live["product_name"] == "AWS VPC Security")):
        part = live[keep]
        report["context"][name] = {count: _auc(part["y"], -part[count]) for count in ("past_pair", "past_src", "past_dst")}
    return report


def system() -> dict:
    """The I19 run: routing and confidence of v2's malicious rows, and v2_incident outcomes by source."""

    report = {}
    for group in ("v2_live", "v2_incident"):
        rows = pd.read_parquet(RESULT_DIR / f"{group}_rows.parquet")
        malicious = rows[rows["label"] == "malicious"]
        report[group] = {"malicious_rows": int(len(malicious)), "routed": int(malicious["routed"].sum()),
                         "confidence_min": round(float(malicious["conf"].min()), 3),
                         "confidence_p05": round(float(malicious["conf"].quantile(0.05)), 3),
                         "final_by_source": {source: part["final"].value_counts().to_dict()
                                             for source, part in malicious.groupby("source")}}
    return report


def leads() -> dict:
    """Leads per incident, identical non-lead twins of each lead, and the bursts behind incidents with ~100 leads."""

    live = pd.read_parquet(GROUPS["v2_live"], columns=COLUMNS + ["timestamp", "incident_ids"])
    threats = live[live["label_binary"] != "benign"].reset_index(drop=True).astype({c: object for c in COLUMNS})
    text = threats[["pipeline", "vendor_name", "product_name"]].fillna("").astype(str)
    threats["view"] = text["pipeline"] + "|" + text["vendor_name"] + "|" + text["product_name"] + "|" + llm_messages(threats)
    malicious = threats["label_binary"].eq("malicious").to_numpy()
    suspicious = threats.loc[~malicious]
    twins = suspicious.groupby("view").size()
    per_row = threats.loc[malicious, "view"].map(twins).fillna(0)
    linked = threats.loc[malicious].assign(incident=lambda d: d["incident_ids"].map(json.loads)).explode("incident")
    per_incident = linked.groupby("incident").size()
    counts = per_incident.value_counts()
    between = lambda low, high: int(counts[(counts.index >= low) & (counts.index <= high)].sum())
    report = {
        "incidents": int(len(per_incident)),
        "leads_per_incident": {"1": between(1, 1), "2": between(2, 2), "3-98": between(3, 98), "99": between(99, 99),
                               "100": between(100, 100), "over 100": between(101, 10**9)},
        "malicious_rows_with_identical_suspicious_twin": round(float((per_row > 0).mean()), 3),
        "identical_suspicious_twins_per_malicious_row_median": float(per_row.median()),
        "busiest_lead_activities": [
            {"view": view[:220], "malicious": int(group["label_binary"].eq("malicious").sum()),
             "suspicious": int(twins.get(view, 0))}
            for view, group in threats[threats["view"].isin(per_row.nlargest(3).index.map(threats.loc[malicious, "view"].get))]
            .groupby("view")],
    }
    # Incidents holding about 100 leads: their leads come from one short burst; count the records of the same
    # source and destination inside that burst that were not attached.
    spans, left = [], []
    for incident in per_incident[per_incident.between(99, 100)].index:
        members = linked[linked["incident"] == incident]
        low, high = members["timestamp"].min(), members["timestamp"].max()
        pairs = set(zip(members["src_ip"], members["dst_ip"]))
        window = suspicious[suspicious["timestamp"].between(low, high)]
        spans.append(high - low)
        left.append(sum(pair in pairs for pair in zip(window["src_ip"], window["dst_ip"])))
    report["incidents_with_99_or_100_leads"] = {"incidents": len(spans),
                                                 "lead_span_seconds_median": round(float(np.median(spans)), 1),
                                                 "same_pair_suspicious_inside_span_median": float(np.median(left))}
    return report


PARTS = ("labels", "same_events", "swap", "separable", "system", "leads")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", default=",".join(PARTS))
    parts = set(parser.parse_args().parts.split(","))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() and parts != set(PARTS) else {}

    if "labels" in parts:
        summary["labels"] = labels()
        for group in ("train", "v4", "latest", "v2_live", "v2_incident"):
            print(f"== {group}")
            for label, entry in summary["labels"][group].items():
                print(f"   {label:10s} {entry}")
        print("   v2 live incidents:", summary["labels"]["v2_live_incidents"])

    if parts & {"same_events", "swap"}:
        pairs_report, old, new = same_events()
        summary["same_events"] = pairs_report
        print("\n== same events:", {k: v for k, v in pairs_report.items() if k != "example"})
        print("   v4:", pairs_report["example"]["v4"])
        print("   v2:", pairs_report["example"]["v2"])
        if "swap" in parts:
            models = load_models()
            summary["swap"] = swap(models, old, new)
            print("\n== swap (paired events)")
            for name, entry in summary["swap"]["pairs"].items():
                print(f"   {name:34s} {entry}")
            for name, entry in summary["swap"]["threat_rows"].items():
                print(f"   {name:24s} {entry}")
            if getattr(models[0], "net", None) is not None:
                import torch

                models[0].net.to("cpu")
                torch.cuda.empty_cache()

    if "separable" in parts:
        summary["separable"] = separable()
        print("\n== separable")
        for view, entry in summary["separable"]["views"].items():
            print(f"   view {view}: {entry}")
        print("   ASA source zone:", summary["separable"]["asa_source_zone"])
        for split, views in summary["separable"]["in_domain"].items():
            for view, entry in views.items():
                print(f"   {split:22s} {view:4s} {entry}")
        print("   past-only context AUC:", summary["separable"]["context"])

    if "system" in parts:
        summary["system"] = system()
        print("\n== I19 system:", summary["system"])

    if "leads" in parts:
        summary["leads"] = leads()
        print("\n== leads:", json.dumps(summary["leads"], indent=1, ensure_ascii=False))
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
