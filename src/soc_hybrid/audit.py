"""Leakage audit from train labels (and unlabeled valid inputs for the shift).

python -m soc_hybrid.audit  →  artifacts/hybrid/audit.json
"""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pandas as pd

from .data import OUT, TRAIN_PATH, VALID_PATH, events, unique_documents
from .text import mask_for_llm, normalize_for_classifier

_NAMESPACE = re.compile(r"\b(USER|ORG|CRED|HOST)-(\d{4})-\d+")
_RESIDUAL_TIME = re.compile(r"(?<![\d.:])\d{1,2}:\d{2}:\d{2}(?!\d)|\b20[12]\d\b")
_MONTH_WORDS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _time_audit(frame: pd.DataFrame) -> dict:
    ts = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    date = ts.dt.strftime("%Y-%m-%d")
    per_label = {}
    for label, rows in frame.groupby("label_binary"):
        d = date[rows.index]
        h = ts[rows.index].dt.hour
        per_label[label] = {
            "rows": int(len(rows)), "first_date": d.min(), "last_date": d.max(), "distinct_dates": int(d.nunique()),
            "share_on_2024_07_26": float((d == "2024-07-26").mean()), "top_hour_share": float(h.value_counts(normalize=True).iloc[0]),
        }
    malicious = frame["label_binary"] == "malicious"
    return {
        "per_label": per_label,
        "malicious_latest": str(ts[malicious].max()),
        "others_earliest": str(ts[~malicious].min()),
        "cutoff_separates_malicious": bool(ts[malicious].max() < ts[~malicious].min()),
    }


def _event_id_audit(frame: pd.DataFrame) -> dict:
    number = frame["event_id"].str.extract(r"(\d+)")[0].astype(int)
    ranges = number.groupby(frame["label_binary"]).agg(["min", "max"])
    return {label: {"min": int(r["min"]), "max": int(r["max"])} for label, r in ranges.iterrows()}


def _pseudonym_audit(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    result = {}
    for token in ("USER-9564", "USER-9546"):
        pattern = re.compile(re.escape(token) + r"(?![-\d])")
        hit = train["message_sanitized"].str.contains(pattern)
        result[f"standalone_{token}_train_rows_by_label"] = train.loc[hit, "label_binary"].value_counts().to_dict()
        result[f"standalone_{token}_valid_rows"] = int(valid["message_sanitized"].str.contains(pattern).sum())

    def namespaces(frame):
        sample = frame["message_sanitized"].sample(n=min(300_000, len(frame)), random_state=0)
        found = sample.map(lambda text: sorted({f"{m.group(1)}-{m.group(2)}" for m in _NAMESPACE.finditer(text)}))
        return found.explode().value_counts().head(6).to_dict()

    result["top_namespaces_train_sample"] = namespaces(train)
    result["top_namespaces_valid_sample"] = namespaces(valid)
    return result


def _perturb(text: str, rng: random.Random) -> str:
    """Rewrite every digit and every month/weekday word: a time-shifted copy of the same event."""

    text = re.sub(r"\d", lambda m: str(rng.randrange(10)), text)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", lambda m: rng.choice(_MONTH_WORDS), text)
    return text


def _invariance(messages: pd.Series, seed: int = 0) -> dict:
    rng = random.Random(seed)
    changed = sum(normalize_for_classifier(m) != normalize_for_classifier(_perturb(m, rng)) for m in messages)
    residual = sum(bool(_RESIDUAL_TIME.search(mask_for_llm(m))) for m in messages)
    return {"messages": int(len(messages)), "classifier_doc_changed": int(changed),
            "llm_view_with_residual_time": int(residual)}


def main() -> None:
    columns = ["event_id", "timestamp", "message_sanitized"]
    train = pd.read_parquet(TRAIN_PATH, columns=columns + ["label_binary"])
    valid = pd.read_parquet(VALID_PATH, columns=columns)
    tr, va = events("train"), events("valid")
    unique = unique_documents(tr)
    train_docs = set(unique["doc"])
    valid_ts = pd.to_datetime(valid["timestamp"], unit="s", utc=True)
    sample = pd.concat([train["message_sanitized"].drop_duplicates().sample(10_000, random_state=1),
                        valid["message_sanitized"].drop_duplicates().sample(10_000, random_state=1)])
    report = {
        "timestamp": _time_audit(train),
        "event_id_ranges": _event_id_audit(train),
        "pseudonyms": _pseudonym_audit(train, valid),
        "invariance_20k_distinct_messages": _invariance(sample),
        "unique_documents": {
            "train_rows": int(len(tr)), "train_unique_docs": int(len(unique)),
            "train_docs_with_two_labels": int((unique.groupby("doc")["label"].nunique() > 1).sum()),
            "valid_rows": int(len(va)), "valid_unique_docs": int(va["doc"].nunique()),
            "valid_unique_docs_seen_in_train": float(pd.Series(va["doc"].unique()).isin(train_docs).mean()),
            "valid_rows_seen_in_train": float(va["doc"].isin(train_docs).mean()),
        },
        "valid_input_shift": {
            "share_on_2024_07_26_to_08_01": float(((valid_ts >= "2024-07-26") & (valid_ts < "2024-08-02")).mean()),
            "valid_sources_unseen_in_train": sorted(set(va["source"]) - set(tr["source"])),
            "source_share": {
                "train": tr["source"].value_counts(normalize=True).round(4).head(6).to_dict(),
                "valid": va["source"].value_counts(normalize=True).round(4).head(6).to_dict(),
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
