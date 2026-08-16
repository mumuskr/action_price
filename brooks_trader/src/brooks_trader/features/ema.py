"""Causal exponential-moving-average calculations."""

import pandas as pd


def calculate_ema20(closes: pd.Series, *, period: int) -> pd.Series:
    """Return the configured causal EMA, whose initial value is the first close.

    The public name follows the Phase 2 ``ema20`` data contract. ``period`` is still
    required so the computational approximation remains configuration-driven.
    """
    if period < 1:
        raise ValueError("period must be at least 1")
    numeric = pd.to_numeric(closes, errors="raise").astype(float)
    if numeric.empty:
        raise ValueError("closes must contain at least one value")
    return numeric.ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_ema_slope(ema: pd.Series, *, lookback: int) -> pd.Series:
    """Return EMA price change per bar over a backward-only lookback."""
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    numeric = pd.to_numeric(ema, errors="raise").astype(float)
    return (numeric - numeric.shift(lookback)) / lookback
