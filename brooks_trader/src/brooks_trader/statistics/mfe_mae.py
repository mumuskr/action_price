"""Direction-aware maximum favorable/adverse excursion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from brooks_trader.models import Bar, Direction


@dataclass(frozen=True)
class Excursion:
    """Price and R-normalized excursion for one completed holding period."""

    mfe: float
    mae: float
    mfe_r: float
    mae_r: float


def calculate_mfe_mae(
    *,
    direction: Direction,
    entry_price: float,
    initial_risk_points: float,
    bars: Sequence[Bar],
) -> Excursion:
    """Calculate excursions from only the bars belonging to an opened trade."""
    if initial_risk_points <= 0:
        raise ValueError("initial_risk_points must be positive")
    if not bars:
        raise ValueError("at least one holding-period bar is required")
    if direction == Direction.LONG:
        mfe = max(0.0, max(bar.high for bar in bars) - entry_price)
        mae = max(0.0, entry_price - min(bar.low for bar in bars))
    else:
        mfe = max(0.0, entry_price - min(bar.low for bar in bars))
        mae = max(0.0, max(bar.high for bar in bars) - entry_price)
    return Excursion(
        mfe=mfe,
        mae=mae,
        mfe_r=mfe / initial_risk_points,
        mae_r=mae / initial_risk_points,
    )
