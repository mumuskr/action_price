from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from brooks_trader.models import Bar, SwingPoint, SwingType


def test_bar_normalizes_timestamp_to_utc() -> None:
    bar = Bar(
        timestamp=datetime(2026, 1, 2, 9, 30, tzinfo=timezone(timedelta(hours=8))),
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1000.0,
    )

    assert bar.timestamp == datetime(2026, 1, 2, 1, 30, tzinfo=UTC)


def test_bar_rejects_invalid_ohlc_relationships() -> None:
    with pytest.raises(ValidationError, match="high must be greater"):
        Bar(
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            open=100.0,
            high=99.0,
            low=98.0,
            close=100.0,
            volume=10.0,
        )


def test_swing_cannot_be_confirmed_before_it_occurs() -> None:
    with pytest.raises(ValidationError, match="confirmed_at cannot precede"):
        SwingPoint(
            index=10,
            swing_time=datetime(2026, 1, 2, 10, tzinfo=UTC),
            price=101.0,
            type=SwingType.HIGH,
            confirmed_at=9,
            confirmation_time=datetime(2026, 1, 2, 10, tzinfo=UTC),
        )
