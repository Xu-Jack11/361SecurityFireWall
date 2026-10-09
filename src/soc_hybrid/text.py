"""Leakage-safe text views of an event.

Two views, both computed per distinct message and broadcast back (logs repeat
heavily):

- ``classifier_documents``: what the classifier sees. Every sanitizer
  pseudonym becomes its kind, every digit run becomes ``0`` and calendar words
  are masked, so no date, time, epoch or year can survive. In train the
  timestamp alone separates malicious from the rest, and the sanitizer even
  rewrote years into pseudonyms (standalone ``USER-9564`` sits on 51,756
  malicious rows and 385 benign ones), so leakage control has to hold by
  construction: a leaking feature looks *better* on any split of train.
- ``llm_messages``: what the LLM reads. Only times and pseudonyms are masked;
  ports, event codes and other numbers keep their meaning.
"""

from __future__ import annotations

import ipaddress
import re

import numpy as np
import pandas as pd

KINDS = ("USER", "ORG", "CRED", "HOST")
# The sanitizer glues tokens to their neighbours ("17CRED-CRED-28950119",
# "USER-CRED-30678MAC"), so no word boundary in front; chains of kinds count as one.
_PSEUDONYM = re.compile(r"(USER|ORG|CRED|HOST)(?:-(?:USER|ORG|CRED|HOST))*(?:-\d+)+")
_LOWER_PSEUDONYM = re.compile(r"\b(user|domain)-\d+")
_MONTHS = (
    "January|February|March|April|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
# A calendar word may be glued to digits ("Jul20" in ps output) but not to letters ("Market").
_CALENDAR = re.compile(rf"(?<![A-Za-z])(?:({_MONTHS})|({_WEEKDAYS}))(?![a-z])", re.IGNORECASE)
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX = re.compile(r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")

# LLM view: times only. Applied in order; dates first, since a time pattern
# must not eat "10.0.0.1:443" or the next CSV field.
_LLM_TIME_MASKS = (
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"), "<DATE>"),
    (re.compile(r"<(?:USER|ORG|CRED|HOST)>[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"), "<DATE>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "<DATE>"),
    (re.compile(rf"\b\d{{1,2}}/(?:{_MONTHS})/(?:\d{{2,4}}|<(?:USER|ORG|CRED|HOST)>)(?::\d{{2}}:\d{{2}}:\d{{2}})?", re.IGNORECASE), "<DATE>"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b", re.IGNORECASE), "<DATE>"),
    (re.compile(rf"(?<![A-Za-z])(?:{_MONTHS})\d{{1,2}}\b"), "<DATE>"),
    (re.compile(rf"\b(?:{_WEEKDAYS})\b"), "<DAY>"),
    (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3,6}(?![\d.]))?(?:\s?[AP]M)?(?:Z|\s?[+-]\d{2}:?\d{2})?(?!\d)"), "<TIME>"),
    (re.compile(r"\b1[4-8]\d{8}(?:\d{3})?(?:\.\d+)?\b"), "<EPOCH>"),
    (re.compile(r"\b20[12]\d\b"), "<YEAR>"),
)


_PARALLEL_MIN = 20_000


def _map_chunk(args):
    function, chunk = args
    return [function(value) for value in chunk]


def _per_distinct(values: pd.Series, function) -> pd.Series:
    """Apply ``function`` once per distinct value (in worker processes when there are many)."""

    codes, uniques = pd.factorize(values.fillna("").astype(str))
    if len(uniques) >= _PARALLEL_MIN:
        import multiprocessing as mp

        workers = min(32, mp.cpu_count())
        chunks = np.array_split(np.asarray(uniques, dtype=object), workers * 4)
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_map_chunk, [(function, list(chunk)) for chunk in chunks])
        mapped = np.array([value for part in parts for value in part], dtype=object)
    else:
        mapped = np.array([function(value) for value in uniques], dtype=object)
    return pd.Series(mapped[codes] if len(uniques) else np.array([], dtype=object), index=values.index, dtype=object)


def normalize_for_classifier(text: str) -> str:
    """Pseudonyms → kind, calendar words → zmon/zday, hex → zhex, digit runs → 0."""

    text = _PSEUDONYM.sub(lambda m: f" z{m.group(1).lower()} ", text)
    text = _LOWER_PSEUDONYM.sub(lambda m: f" z{m.group(1)} ", text)
    text = _CALENDAR.sub(lambda m: " zmon " if m.group(1) else " zday ", text)
    text = _UUID.sub(" zuuid ", text)
    text = _HEX.sub(" zhex ", text)
    text = _DIGITS.sub("0", text)
    return _SPACES.sub(" ", text).strip()


def mask_for_llm(text: str) -> str:
    """Pseudonyms → <KIND>, dates/times/epochs/years → placeholders; other numbers stay."""

    text = _PSEUDONYM.sub(lambda m: f"<{m.group(1)}>", text)
    text = _LOWER_PSEUDONYM.sub(lambda m: f"<{m.group(1).upper()}>", text)
    for pattern, replacement in _LLM_TIME_MASKS:
        text = pattern.sub(replacement, text)
    return text.strip()


def ip_kind(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "none"
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return "other"
    if address in ipaddress.ip_network("100.64.0.0/10"):
        return "cgnat"
    if address.is_private:
        return "private"
    return "public"


def port_bucket(value: str) -> str:
    value = (value or "").strip()
    if not value.isdigit():
        return "none"
    port = int(value)
    return "wellknown" if port < 1024 else "registered" if port < 49152 else "ephemeral"


def _token(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", (value or "").strip().lower()).strip("_")
    return value or "none"


def source_keys(frame: pd.DataFrame) -> pd.Series:
    """vendor/product, the log source an event came from."""

    vendor = frame["vendor_name"].fillna("").astype(str).str.strip()
    product = frame["product_name"].fillna("").astype(str).str.strip()
    return vendor + "/" + product


def classifier_documents(frame: pd.DataFrame) -> pd.Series:
    """One document per event: source tokens, field shapes and the normalized message."""

    def column(name: str) -> pd.Series:
        return frame[name].fillna("").astype(str) if name in frame.columns else pd.Series("", index=frame.index)

    vendor = _per_distinct(column("vendor_name"), _token)
    product = _per_distinct(column("product_name"), _token)
    pipeline = _per_distinct(column("pipeline"), _token)
    fields = (
        "fpipe_" + pipeline
        + " fvendor_" + vendor
        + " fproduct_" + product
        + " fsrcip_" + _per_distinct(column("src_ip"), ip_kind)
        + " fdstip_" + _per_distinct(column("dst_ip"), ip_kind)
        + " fsrcport_" + _per_distinct(column("src_port"), port_bucket)
        + " fsrchost_" + column("src_host").str.strip().ne("").map({True: "yes", False: "no"})
        + " fdsthost_" + column("dst_host").str.strip().ne("").map({True: "yes", False: "no"})
        + " fuser_" + column("username").str.strip().ne("").map({True: "yes", False: "no"})
    )
    message = _per_distinct(column("message_sanitized"), normalize_for_classifier)
    return fields + " | " + message


def llm_messages(frame: pd.DataFrame) -> pd.Series:
    return _per_distinct(frame["message_sanitized"], mask_for_llm)


def raw_documents(frame: pd.DataFrame) -> pd.Series:
    """Ablation only: raw fields, raw message and hour/weekday/month tokens, i.e. no leakage control."""

    ts = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    time_tokens = ("thour_" + ts.dt.hour.astype(str) + " tdow_" + ts.dt.dayofweek.astype(str)
                   + " tmonth_" + ts.dt.month.astype(str))
    fields = pd.Series("", index=frame.index, dtype=object)
    for name in ("pipeline", "vendor_name", "product_name", "src_ip", "dst_ip", "src_port", "src_host", "dst_host", "username"):
        fields = fields + " f" + name + "_" + frame[name].fillna("").astype(str).str.replace(r"\s+", "_", regex=True)
    return time_tokens + fields + " | " + frame["message_sanitized"].fillna("").astype(str)
