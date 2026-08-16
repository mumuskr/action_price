"""Strict local OHLCV loading without feature calculation or future-data access."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from brooks_trader.models.bar import Bar

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
TimestampUnit = Literal["s", "ms", "us", "ns"]


class OHLCVValidationError(ValueError):
    """Raised when a historical file violates the canonical OHLCV contract."""


def read_parquet_frame(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file with bounded, deterministic Arrow resource usage.

    Parallel pre-buffering is unnecessary for the local research partitions and has caused
    native Arrow allocator crashes on macOS. Keeping both options disabled also prevents a
    Dashboard request from creating a large background reader pool.
    """
    return pd.read_parquet(
        Path(path).expanduser(),
        engine="pyarrow",
        use_threads=False,
        pre_buffer=False,
    )


def load_ohlcv(
    path: str | Path,
    *,
    timestamp_unit: TimestampUnit | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load CSV or Parquet data and return validated canonical OHLCV columns.

    String timestamps without an offset are interpreted as UTC. Integer epoch timestamps
    require an explicit ``timestamp_unit`` so a unit cannot be guessed incorrectly.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"OHLCV file does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        frame = read_parquet_frame(source)
    else:
        raise ValueError(f"unsupported OHLCV format {suffix!r}; expected .csv or .parquet")

    if column_map:
        frame = frame.rename(columns=dict(column_map))
    return normalize_ohlcv(frame, timestamp_unit=timestamp_unit)


def normalize_ohlcv(
    frame: pd.DataFrame,
    *,
    timestamp_unit: TimestampUnit | None = None,
) -> pd.DataFrame:
    """Normalize and validate an in-memory OHLCV frame.

    The returned frame is sorted by timestamp, has a zero-based index, and contains only
    canonical columns. Duplicate timestamps are rejected rather than silently aggregated.
    """
    if frame.empty:
        raise OHLCVValidationError("OHLCV data must contain at least one row")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise OHLCVValidationError(f"duplicate columns after normalization: {duplicates}")

    missing = [column for column in OHLCV_COLUMNS if column not in normalized.columns]
    if missing:
        raise OHLCVValidationError(f"missing required OHLCV columns: {missing}")
    normalized = normalized.loc[:, list(OHLCV_COLUMNS)].copy()

    normalized["timestamp"] = _parse_timestamps(
        normalized["timestamp"], timestamp_unit=timestamp_unit
    )
    invalid_timestamps = normalized["timestamp"].isna()
    if invalid_timestamps.any():
        rows = normalized.index[invalid_timestamps].tolist()[:10]
        raise OHLCVValidationError(f"invalid timestamps at source rows: {rows}")

    for column in OHLCV_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    numeric_values = normalized.loc[:, list(OHLCV_COLUMNS[1:])].to_numpy(dtype=float)
    invalid_numeric = ~np.isfinite(numeric_values)
    if invalid_numeric.any():
        row_offsets, column_offsets = np.where(invalid_numeric)
        examples = [
            (normalized.index[row], OHLCV_COLUMNS[column + 1])
            for row, column in zip(row_offsets[:10], column_offsets[:10], strict=True)
        ]
        raise OHLCVValidationError(f"non-finite or non-numeric OHLCV values at: {examples}")

    _validate_price_relationships(normalized)
    if (normalized["volume"] < 0).any():
        rows = normalized.index[normalized["volume"] < 0].tolist()[:10]
        raise OHLCVValidationError(f"volume cannot be negative at source rows: {rows}")

    if normalized["timestamp"].duplicated().any():
        duplicates = (
            normalized.loc[normalized["timestamp"].duplicated(keep=False), "timestamp"]
            .astype(str)
            .unique()
            .tolist()[:10]
        )
        raise OHLCVValidationError(f"duplicate timestamps are not allowed: {duplicates}")

    return normalized.sort_values("timestamp", kind="stable").reset_index(drop=True)


def bars_from_frame(frame: pd.DataFrame) -> list[Bar]:
    """Convert a normalized or raw frame to immutable ``Bar`` models."""
    normalized = normalize_ohlcv(frame)
    return [Bar.model_validate(record) for record in normalized.to_dict(orient="records")]


def _parse_timestamps(
    values: pd.Series,
    *,
    timestamp_unit: TimestampUnit | None,
) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        if timestamp_unit is None:
            raise OHLCVValidationError(
                "numeric timestamps require timestamp_unit ('s', 'ms', 'us', or 'ns')"
            )
        return pd.to_datetime(values, unit=timestamp_unit, errors="coerce", utc=True)
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True)


def _validate_price_relationships(frame: pd.DataFrame) -> None:
    invalid_high_low = frame["high"] < frame["low"]
    invalid_high = frame["high"] < frame[["open", "close"]].max(axis=1)
    invalid_low = frame["low"] > frame[["open", "close"]].min(axis=1)
    invalid = invalid_high_low | invalid_high | invalid_low
    if invalid.any():
        rows = frame.index[invalid].tolist()[:10]
        raise OHLCVValidationError(f"invalid OHLC price relationships at source rows: {rows}")
