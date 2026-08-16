"""Strategy signal contract kept separate from setups and execution intents."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import Direction, DomainModel
from brooks_trader.models.setup import TradeSetup


class SignalType(StrEnum):
    SECOND_ENTRY_WITH_TREND = "SECOND_ENTRY_WITH_TREND"


class StrategySignal(DomainModel):
    """A Trading Engine signal without quantity, order state, or broker authority."""

    signal_type: SignalType
    created_at: datetime
    signal_bar_index: int = Field(ge=0)
    direction: Direction
    setup: TradeSetup
    reasons: list[str] = Field(default_factory=list)
    strategy_version: str = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_signal_consistency(self) -> "StrategySignal":
        if self.direction != self.setup.direction:
            raise ValueError("signal and setup directions must match")
        if self.signal_bar_index != self.setup.signal_bar_index:
            raise ValueError("signal and setup indices must match")
        if self.strategy_version != self.setup.strategy_version:
            raise ValueError("signal and setup strategy versions must match")
        return self
