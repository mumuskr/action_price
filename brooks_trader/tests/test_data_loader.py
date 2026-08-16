from pathlib import Path

import pandas as pd
import pytest

from brooks_trader.data.loader import (
    OHLCVValidationError,
    load_ohlcv,
    normalize_ohlcv,
    read_parquet_frame,
)


def test_parquet_reader_disables_arrow_threads_and_prebuffer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "bars.parquet"
    expected = pd.DataFrame({"close": [100.5]})
    observed: dict[str, object] = {}

    def fake_read_parquet(path: Path, **kwargs: object) -> pd.DataFrame:
        observed["path"] = path
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    result = read_parquet_frame(source)

    assert result is expected
    assert observed == {
        "path": source,
        "engine": "pyarrow",
        "use_threads": False,
        "pre_buffer": False,
    }


def test_normalize_ohlcv_sorts_and_canonicalizes_columns() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": ["2026-01-02T09:31:00+08:00", "2026-01-02T09:30:00+08:00"],
            "Open": [101.0, 100.0],
            "High": [102.0, 101.0],
            "Low": [100.0, 99.0],
            "Close": [101.5, 100.5],
            "Volume": [20, 10],
            "Ignored": [1, 2],
        }
    )

    result = normalize_ohlcv(raw)

    assert result.columns.tolist() == ["timestamp", "open", "high", "low", "close", "volume"]
    assert result["open"].tolist() == [100.0, 101.0]
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_load_ohlcv_reads_csv(tmp_path) -> None:
    source = tmp_path / "bars.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-02T09:30:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
        }
    ).to_csv(source, index=False)

    result = load_ohlcv(source)

    assert len(result) == 1
    assert result.iloc[0]["close"] == 100.5


def test_numeric_timestamps_require_explicit_unit() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [1_767_225_600_000],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
        }
    )

    with pytest.raises(OHLCVValidationError, match="timestamp_unit"):
        normalize_ohlcv(frame)


def test_duplicate_timestamps_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T09:30:00Z", "2026-01-02T09:30:00Z"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "volume": [10, 10],
        }
    )

    with pytest.raises(OHLCVValidationError, match="duplicate timestamps"):
        normalize_ohlcv(frame)


def test_invalid_price_relationships_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T09:30:00Z"],
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [100.5],
            "volume": [10],
        }
    )

    with pytest.raises(OHLCVValidationError, match="price relationships"):
        normalize_ohlcv(frame)
