"""US regular-session calendar validation and deterministic bar aggregation."""

from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

from brooks_trader.data.loader import normalize_ohlcv

SUPPORTED_MINUTES = (5, 15, 30)


def nyse_calendar(start: date, end: date) -> xcals.ExchangeCalendar:
    """Build the XNYS calendar for the exact research interval."""
    return xcals.get_calendar("XNYS", start=start, end=end + timedelta(days=1), side="left")


def validate_regular_session_minutes(
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> None:
    """Require exactly one bar for every XNYS regular-session minute."""
    missing, unexpected = regular_session_minute_differences(frame, start=start, end=end)
    if missing.empty and unexpected.empty:
        return
    raise ValueError(
        "minute coverage mismatch: "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"first_missing={missing[0].isoformat() if len(missing) else None} "
        f"first_unexpected={unexpected[0].isoformat() if len(unexpected) else None}"
    )


def regular_session_minute_differences(
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return expected XNYS minutes missing from data and unexpected timestamps."""
    normalized = normalize_ohlcv(frame)
    calendar = nyse_calendar(start, end)
    sessions = calendar.sessions_in_range(start, end)
    expected = pd.DatetimeIndex(
        pd.concat(
            [calendar.session_minutes(session).to_series(index=None) for session in sessions],
            ignore_index=True,
        )
    )
    actual = pd.DatetimeIndex(normalized["timestamp"])
    return expected.difference(actual), actual.difference(expected)


def resample_regular_session_bars(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate one-minute bars within each New York trading session."""
    if minutes not in SUPPORTED_MINUTES:
        raise ValueError(f"minutes must be one of {SUPPORTED_MINUTES}")

    normalized = normalize_ohlcv(frame)
    indexed = normalized.set_index("timestamp")
    session_dates = indexed.index.tz_convert("America/New_York").date
    aggregated: list[pd.DataFrame] = []
    for _, session in indexed.groupby(session_dates, sort=True):
        origin = session.index[0]
        result = session.resample(
            f"{minutes}min",
            origin=origin,
            closed="left",
            label="left",
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregated.append(result.dropna(subset=["open", "high", "low", "close"]))

    if not aggregated:
        raise ValueError("cannot resample an empty bar collection")
    combined = pd.concat(aggregated).reset_index()
    return normalize_ohlcv(combined)
