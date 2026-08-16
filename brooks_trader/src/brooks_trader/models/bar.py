"""OHLCV and bar-feature data contracts."""

from datetime import UTC, datetime
from math import isfinite

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import DomainModel


class Bar(DomainModel):
    """A single immutable OHLCV observation with a UTC timestamp."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require an unambiguous timestamp and normalize it to UTC."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("open", "high", "low", "close", "volume")
    @classmethod
    def require_finite_number(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("OHLCV values must be finite")
        return value

    @model_validator(mode="after")
    def validate_price_relationships(self) -> "Bar":
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be greater than or equal to open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be less than or equal to open and close")
        return self


class BarFeatures(DomainModel):
    """A completed Phase 2 bar-feature observation.

    Every value must be derived from the current bar and information available before it.
    """

    timestamp: datetime
    bar_index: int = Field(ge=0)
    range: float = Field(ge=0)
    body: float = Field(ge=0)
    body_ratio: float = Field(ge=0, le=1)
    upper_tail: float = Field(ge=0)
    lower_tail: float = Field(ge=0)
    upper_tail_ratio: float = Field(ge=0, le=1)
    lower_tail_ratio: float = Field(ge=0, le=1)
    close_location: float = Field(ge=0, le=1)
    bull_bar: bool
    bear_bar: bool
    trend_bar: bool
    doji: bool
    inside_bar: bool
    outside_bar: bool
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool
    overlap_previous: float = Field(ge=0, le=1)
    ema20: float | None = None
    distance_to_ema: float | None = None
    ema_slope: float | None = None

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("ema20", "distance_to_ema", "ema_slope")
    @classmethod
    def require_finite_optional_number(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("feature values must be finite when present")
        return value
