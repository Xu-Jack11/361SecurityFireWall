"""Shared constants for the SOC baseline."""

ID_COLUMN = "event_id"
LABEL_COLUMN = "label_binary"
PRED_COLUMN = "pred_label"

LABELS = ("benign", "suspicious", "malicious")
LABEL_CANDIDATES = (LABEL_COLUMN, "label", "target")

FEATURE_COLUMNS = (
    "timestamp",
    "pipeline",
    "src_ip",
    "dst_ip",
    "src_port",
    "src_host",
    "dst_host",
    "username",
    "message_sanitized",
    "product_name",
    "vendor_name",
)

CATEGORICAL_COLUMNS = (
    "pipeline",
    "src_ip",
    "dst_ip",
    "src_host",
    "dst_host",
    "username",
    "product_name",
    "vendor_name",
)

