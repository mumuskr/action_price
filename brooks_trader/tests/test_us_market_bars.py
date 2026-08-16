from datetime import UTC, datetime

import pandas as pd
import pytest

from brooks_trader.data.us_market_bars import (
    regular_session_minute_differences,
    resample_regular_session_bars,
    validate_regular_session_minutes,
)


def test_resample_regular_session_bars_aggregates_ohlcv() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                datetime(2026, 8, 7, 13, 30, tzinfo=UTC), periods=5, freq="1min"
            ),
            "open": [10.0, 10.5, 11.0, 10.75, 11.25],
            "high": [10.6, 11.1, 11.2, 11.3, 11.5],
            "low": [9.9, 10.4, 10.7, 10.6, 11.0],
            "close": [10.5, 11.0, 10.75, 11.25, 11.4],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    result = resample_regular_session_bars(frame, 5)

    assert len(result) == 1
    assert result.iloc[0][["open", "high", "low", "close", "volume"]].to_dict() == {
        "open": 10.0,
        "high": 11.5,
        "low": 9.9,
        "close": 11.4,
        "volume": 150.0,
    }


def test_regular_session_validation_detects_missing_minute() -> None:
    timestamps = pd.date_range(datetime(2026, 8, 7, 13, 30, tzinfo=UTC), periods=389, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1,
        }
    )

    with pytest.raises(ValueError, match="missing=1"):
        validate_regular_session_minutes(
            frame,
            start=timestamps[0].date(),
            end=timestamps[0].date(),
        )

    missing, unexpected = regular_session_minute_differences(
        frame,
        start=timestamps[0].date(),
        end=timestamps[0].date(),
    )
    assert len(missing) == 1
    assert unexpected.empty
