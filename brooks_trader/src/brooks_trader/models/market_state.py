"""Market-context models kept separate from patterns and trade decisions."""

from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from pydantic import Field, field_validator

from brooks_trader.models.common import DomainModel


class MarketRegime(StrEnum):
    STRONG_BULL_TREND = "STRONG_BULL_TREND"
    BULL_TREND = "BULL_TREND"
    WEAK_BULL = "WEAK_BULL"
    TRADING_RANGE = "TRADING_RANGE"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR_TREND = "BEAR_TREND"
    STRONG_BEAR_TREND = "STRONG_BEAR_TREND"


class AlwaysInState(StrEnum):
    ALWAYS_IN_LONG = "ALWAYS_IN_LONG"
    ALWAYS_IN_SHORT = "ALWAYS_IN_SHORT"
    NEUTRAL = "NEUTRAL"


class MarketState(DomainModel):
    """A point-in-time computational proxy for Brooks-style market context."""

    timestamp: datetime
    bar_index: int = Field(ge=0)
    regime: MarketRegime
    always_in: AlwaysInState = AlwaysInState.NEUTRAL
    trend_score: float = Field(ge=-1, le=1)
    ema_score: float = Field(default=0.0, ge=-1, le=1)
    structure_score: float = Field(default=0.0, ge=-1, le=1)
    pressure_score: float = Field(default=0.0, ge=-1, le=1)
    overlap_score: float = Field(default=0.0, ge=-1, le=1)
    breakout_score: float = Field(default=0.0, ge=-1, le=1)
    strategy_version: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator(
        "trend_score",
        "ema_score",
        "structure_score",
        "pressure_score",
        "overlap_score",
        "breakout_score",
    )
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("market-state scores must be finite")
        return value
