"""Load the parquet inputs once, build both text views, and cache them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text import classifier_documents, llm_messages, source_keys

TRAIN_PATH = Path("data/train.parquet")
VALID_PATH = Path("data/valid_input.parquet")
ANSWER_PATH = Path("data/valid_answer_private.parquet")
OUT = Path("artifacts/hybrid")
CACHE = OUT / "cache"
LABELS = ("benign", "suspicious", "malicious")
ALERTS = ("suspicious", "malicious")
COLUMNS = ["event_id", "pipeline", "vendor_name", "product_name", "src_ip", "dst_ip", "src_port",
           "src_host", "dst_host", "username", "message_sanitized"]


def build_events(path: Path, label: bool) -> pd.DataFrame:
    """event_id, source, classifier doc, LLM message (and label): no timestamp, no raw ids."""

    frame = pd.read_parquet(path, columns=COLUMNS + (["label_binary"] if label else []))
    events = pd.DataFrame({
        "event_id": frame["event_id"].astype(str),
        "source": source_keys(frame),
        "doc": classifier_documents(frame),
        "llm_message": llm_messages(frame),
        "pipeline": frame["pipeline"].fillna("").astype(str),
        "vendor_name": frame["vendor_name"].fillna("").astype(str),
        "product_name": frame["product_name"].fillna("").astype(str),
    })
    if label:
        events["label"] = frame["label_binary"].astype(str)
    return events


def events(split: str) -> pd.DataFrame:
    """Cached event table for ``train`` or ``valid`` (valid without labels)."""

    path = CACHE / f"{split}_events.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        table = build_events(TRAIN_PATH if split == "train" else VALID_PATH, label=split == "train")
        table.to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)
    return pd.read_parquet(path)


def valid_answers() -> pd.Series:
    """valid labels by event_id. Only metrics.py and the report scripts may call this."""

    answers = pd.read_parquet(ANSWER_PATH)
    return answers.assign(event_id=answers["event_id"].astype(str)).set_index("event_id")["label_binary"]


def unique_documents(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (doc, label): its source, row count and a representative event."""

    keys = ["doc", "label"] if "label" in table else ["doc"]
    grouped = table.groupby(keys, sort=False)
    unique = grouped.agg(source=("source", "first"), rows=("event_id", "size"), event_id=("event_id", "first"),
                         llm_message=("llm_message", "first"), pipeline=("pipeline", "first"),
                         vendor_name=("vendor_name", "first"), product_name=("product_name", "first"))
    return unique.reset_index()
