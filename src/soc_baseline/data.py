"""Data loading and sampling utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import LABEL_CANDIDATES


def detect_label_column(df: pd.DataFrame) -> str:
    """Find the label column used by the competition training file."""

    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find a label column. Tried: {', '.join(LABEL_CANDIDATES)}")


def read_parquet_frame(path: str | Path, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read a parquet file with a clearer error for missing paths."""

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path, columns=list(columns) if columns else None)


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Return a shuffled stratified sample while keeping every class if possible."""

    if max_rows is None or max_rows >= len(df):
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    counts = df[label_col].value_counts(dropna=False)
    if max_rows < len(counts):
        raise ValueError(
            f"max_rows={max_rows} is smaller than the number of classes={len(counts)}"
        )

    raw_targets = counts / len(df) * max_rows
    targets = raw_targets.apply(int).clip(lower=1)

    while int(targets.sum()) < max_rows:
        room = counts - targets
        eligible = room[room > 0]
        if eligible.empty:
            break
        fractions = (raw_targets - raw_targets.apply(int)).loc[eligible.index]
        targets.loc[fractions.sort_values(ascending=False).index[0]] += 1

    while int(targets.sum()) > max_rows:
        reducible = targets[targets > 1]
        if reducible.empty:
            break
        over_allocated = (targets - raw_targets).loc[reducible.index]
        targets.loc[over_allocated.sort_values(ascending=False).index[0]] -= 1

    pieces = []
    for label_value, target in targets.items():
        group = df[df[label_col] == label_value]
        pieces.append(group.sample(n=int(target), random_state=random_state))

    return pd.concat(pieces, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

