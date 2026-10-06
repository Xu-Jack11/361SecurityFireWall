"""Feature construction for security event logs."""

from __future__ import annotations

import ipaddress
import re
from typing import Any

import pandas as pd

from .constants import CATEGORICAL_COLUMNS

_TOKEN_CLEAN_RE = re.compile(r"[^0-9A-Za-z_.:/=@+-]+")

# "full" is the original baseline document. "content" drops every token that
# memorizes *when* or *who* rather than *what*: in train.parquet benign and
# suspicious rows are all stamped 2024-07-26 ~11:00 UTC while malicious rows
# span 2022-06..2024-07-18, and the test set's hosts, users and sanitizer IDs
# barely overlap with training, so time and identifier tokens do not transfer.
FEATURE_SETS = ("full", "content")

CONTENT_CATEGORICAL_COLUMNS = ("pipeline", "product_name", "vendor_name")
IDENTIFIER_COLUMNS = ("src_ip", "dst_ip", "src_host", "dst_host", "username")

_MONTHS = (
    "January|February|March|April|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
# Applied in order. Sanitizer IDs go first because WitFoo's sanitizer also
# rewrites parts of dates (e.g. "USER-9546-07-26T06:28:52"); IPs go before
# clock times so "outside:10.1.2.3/443" is not mistaken for a time.
_MESSAGE_NORMALIZERS = (
    (re.compile(r"(USER|ORG|CRED|HOST)(?:-\d+)+"), lambda m: f"tok_{m.group(1).lower()}"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?=T|\b)"), "tok_date"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}\b"), "tok_date"),
    (re.compile(rf"\b(?:{_WEEKDAYS})\b"), "tok_weekday"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "tok_ip"),
    (re.compile(r"(?:\b|(?<=T))\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s?[AP]M)?(?:Z|[+-]\d{2}:?\d{2})?"), "tok_time"),
    (re.compile(r"\b1[4-8]\d{8}(?:\d{3})?(?:\.\d+)?\b"), "tok_epoch"),
    (re.compile(r"\b20[12]\d\b"), "tok_year"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "tok_uuid"),
    (re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}\b", re.IGNORECASE), "tok_hex"),
    (re.compile(r"\b\d{5,}\b"), "tok_num"),
)


def port_bucket(value: Any) -> str:
    """Bucket a source port into coarse, stable ranges."""

    if value is None or pd.isna(value):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    try:
        port = int(float(text))
    except ValueError:
        return "invalid"
    if port < 0 or port > 65535:
        return "out_of_range"
    if port <= 1023:
        return "well_known"
    if port <= 49151:
        return "registered"
    return "ephemeral"


def ip_prefix(value: Any) -> str:
    """Return an IPv4 /24-like prefix token or a missing/non-IPv4 marker."""

    if value is None or pd.isna(value):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return "non_ipv4"
    if parsed.version != 4:
        return "non_ipv4"
    return ".".join(text.split(".")[:3])


def value_kind(value: Any) -> str:
    """Coarse shape of an identifier field: missing, ipv4, ipv6 or other."""

    if value is None or pd.isna(value):
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    try:
        return f"ipv{ipaddress.ip_address(text).version}"
    except ValueError:
        return "other"


def normalize_message(messages: pd.Series) -> pd.Series:
    """Replace timestamps, IPs, sanitizer IDs and long numbers with placeholders."""

    # Logs repeat heavily (the test set has ~180k distinct messages in 2M rows),
    # so normalize each distinct message once and broadcast back.
    codes, uniques = pd.factorize(messages.fillna("").astype(str))
    normalized = pd.Series(uniques, dtype="object")
    for pattern, replacement in _MESSAGE_NORMALIZERS:
        normalized = normalized.str.replace(pattern, replacement, regex=True)
    normalized = normalized.str.strip()
    return pd.Series(normalized.to_numpy()[codes], index=messages.index, dtype="object")


def build_log_documents(df: pd.DataFrame, feature_set: str = "full") -> pd.Series:
    """Build one model document per event from structured fields and message text."""

    if feature_set == "content":
        return _build_content_documents(df)
    if feature_set != "full":
        raise ValueError(f"Unknown feature_set {feature_set!r}; expected one of {FEATURE_SETS}")

    docs = pd.Series("", index=df.index, dtype="object")

    for column in CATEGORICAL_COLUMNS:
        values = _clean_token_series(_column_or_empty(df, column))
        docs = docs.str.cat(column + "=" + values, sep=" ")

    src_ports = _column_or_empty(df, "src_port").map(port_bucket)
    docs = docs.str.cat("src_port_bucket=" + src_ports, sep=" ")

    src_prefixes = _column_or_empty(df, "src_ip").map(ip_prefix)
    dst_prefixes = _column_or_empty(df, "dst_ip").map(ip_prefix)
    docs = docs.str.cat("src_ip_prefix=" + src_prefixes, sep=" ")
    docs = docs.str.cat("dst_ip_prefix=" + dst_prefixes, sep=" ")

    timestamp_tokens = _timestamp_tokens(_column_or_empty(df, "timestamp"))
    for column in timestamp_tokens.columns:
        docs = docs.str.cat(column + "=" + timestamp_tokens[column].astype(str), sep=" ")

    messages = _column_or_empty(df, "message_sanitized").fillna("").astype(str).str.strip()
    docs = docs.str.cat(messages, sep=" ")

    return docs.str.replace(r"\s+", " ", regex=True).str.strip()


def _build_content_documents(df: pd.DataFrame) -> pd.Series:
    docs = pd.Series("", index=df.index, dtype="object")

    for column in CONTENT_CATEGORICAL_COLUMNS:
        values = _clean_token_series(_column_or_empty(df, column))
        docs = docs.str.cat(column + "=" + values, sep=" ")

    for column in IDENTIFIER_COLUMNS:
        kinds = _column_or_empty(df, column).map(value_kind)
        docs = docs.str.cat(column + "_kind=" + kinds, sep=" ")

    src_ports = _column_or_empty(df, "src_port").map(port_bucket)
    docs = docs.str.cat("src_port_bucket=" + src_ports, sep=" ")

    messages = normalize_message(_column_or_empty(df, "message_sanitized"))
    docs = docs.str.cat("message_empty=" + (messages == "").map({True: "yes", False: "no"}), sep=" ")
    docs = docs.str.cat(messages, sep=" ")

    return docs.str.replace(r"\s+", " ", regex=True).str.strip()


def _column_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _clean_token_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    values = values.mask(values == "", "missing")
    values = values.str.replace(r"\s+", "_", regex=True)
    values = values.map(lambda value: _TOKEN_CLEAN_RE.sub("_", value).strip("_") or "missing")
    return values


def _timestamp_tokens(series: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    timestamps = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.DataFrame(
        {
            "timestamp_hour": timestamps.dt.hour.fillna(-1).astype(int),
            "timestamp_dayofweek": timestamps.dt.dayofweek.fillna(-1).astype(int),
            "timestamp_month": timestamps.dt.month.fillna(-1).astype(int),
        },
        index=series.index,
    )

