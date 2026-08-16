"""Pattern events produced by stateful detectors."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import Direction, DomainModel
from brooks_trader.models.market_state import MarketState


class PatternType(StrEnum):
    H1 = "H1"
    H2 = "H2"
    L1 = "L1"
    L2 = "L2"


class PatternEvent(DomainModel):
    """A detected pattern, not a setup, signal, or instruction to trade."""

    pattern_type: PatternType
    direction: Direction
    start_index: int = Field(ge=0)
    signal_index: int = Field(ge=0)
    signal_time: datetime
    trigger_price: float = Field(gt=0)
    context: MarketState | None = None
    confidence_score: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_version: str = Field(min_length=1)

    @field_validator("signal_time")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("signal_time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_indices(self) -> "PatternEvent":
        if self.signal_index < self.start_index:
            raise ValueError("signal_index cannot precede start_index")
        return self
