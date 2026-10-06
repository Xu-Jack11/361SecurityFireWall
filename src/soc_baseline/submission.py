"""Competition submission validation and writing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import ID_COLUMN, LABELS, PRED_COLUMN


def validate_submission_frame(
    submission: pd.DataFrame,
    expected_event_ids: pd.Series | Iterable[str],
    allowed_labels: tuple[str, ...] = LABELS,
) -> None:
    """Validate submission shape, IDs, duplicates, and label values."""

    errors: list[str] = []
    expected = pd.Series(list(expected_event_ids), dtype="string")

    required_columns = [ID_COLUMN, PRED_COLUMN]
    if list(submission.columns) != required_columns:
        errors.append(f"columns must be exactly {required_columns}")
        missing_cols = [column for column in required_columns if column not in submission.columns]
        if missing_cols:
            raise ValueError("; ".join(errors))

    event_ids = submission[ID_COLUMN].astype("string")
    labels = submission[PRED_COLUMN].astype("string")

    if event_ids.duplicated().any():
        errors.append("duplicate event_id values found")

    expected_set = set(expected.dropna())
    actual_set = set(event_ids.dropna())
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"missing event_id count={len(missing)} sample={missing[:5]}")
    if extra:
        errors.append(f"extra event_id count={len(extra)} sample={extra[:5]}")

    invalid_labels = sorted(set(labels.dropna()) - set(allowed_labels))
    if invalid_labels:
        errors.append(f"invalid pred_label values={invalid_labels}")
    if labels.isna().any():
        errors.append("pred_label contains null values")

    if len(submission) != len(expected):
        errors.append(f"row count mismatch expected={len(expected)} actual={len(submission)}")

    if errors:
        raise ValueError("; ".join(errors))


def write_submission(
    event_ids: Iterable[str],
    labels: Iterable[str],
    output_path: str | Path,
) -> Path:
    """Write a two-column competition submission CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID_COLUMN: list(event_ids), PRED_COLUMN: list(labels)})
    validate_submission_frame(frame, frame[ID_COLUMN], LABELS)
    frame.to_csv(path, index=False)
    return path

