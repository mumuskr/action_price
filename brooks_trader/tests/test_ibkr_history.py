from datetime import UTC, datetime

import pytest
from ib_async import BarData

from brooks_trader.data.ibkr_history import ibkr_bars_to_frame


def test_ibkr_bars_to_frame_returns_canonical_utc_data() -> None:
    bars = [
        BarData(
            date=datetime(2026, 8, 7, 13, 30, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1_000,
        )
    ]

    result = ibkr_bars_to_frame(bars)

    assert result.columns.tolist() == ["timestamp", "open", "high", "low", "close", "volume"]
    assert result.iloc[0]["timestamp"] == datetime(2026, 8, 7, 13, 30, tzinfo=UTC)
    assert result.iloc[0]["volume"] == 1_000


def test_ibkr_bars_to_frame_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ibkr_bars_to_frame([])
