"""Trading setup models evaluated after context and pattern detection."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import Direction, DomainModel
from brooks_trader.models.market_state import MarketState
from brooks_trader.models.pattern import PatternEvent


class SetupType(StrEnum):
    H2_WITH_TREND = "H2_WITH_TREND"
    L2_WITH_TREND = "L2_WITH_TREND"


class TradeSetup(DomainModel):
    """An evaluated opportunity that remains distinct from a trading signal."""

    setup_type: SetupType
    direction: Direction
    evaluated_at: datetime
    signal_bar_index: int = Field(ge=0)
    pattern: PatternEvent
    market_state: MarketState
    pattern_score: float = Field(ge=0, le=1)
    context_score: float = Field(ge=0, le=1)
    entry: float = Field(gt=0)
    initial_stop: float = Field(gt=0)
    target: float | None = Field(default=None, gt=0)
    invalidation: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_version: str = Field(min_length=1)

    @field_validator("evaluated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_setup_consistency(self) -> "TradeSetup":
        if self.pattern.direction != self.direction:
            raise ValueError("pattern and setup directions must match")
        if self.pattern.signal_index != self.signal_bar_index:
            raise ValueError("pattern and setup signal indices must match")
        if self.pattern.context is not None and self.pattern.context != self.market_state:
            raise ValueError("pattern and setup market states must match")
        if not (
            self.strategy_version
            == self.pattern.strategy_version
            == self.market_state.strategy_version
        ):
            raise ValueError("setup artifacts must share one strategy_version")
        if self.direction == Direction.LONG:
            if self.initial_stop >= self.entry:
                raise ValueError("long setup stop must be below entry")
            if self.target is not None and self.target <= self.entry:
                raise ValueError("long setup target must be above entry")
        else:
            if self.initial_stop <= self.entry:
                raise ValueError("short setup stop must be above entry")
            if self.target is not None and self.target >= self.entry:
                raise ValueError("short setup target must be below entry")
        return self


class SetupEvaluation(DomainModel):
    """Auditable acceptance or rejection of a PatternEvent as a TradeSetup."""

    evaluated_at: datetime
    pattern: PatternEvent
    accepted: bool
    setup: TradeSetup | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_version: str = Field(min_length=1)

    @field_validator("evaluated_at")
    @classmethod
    def normalize_evaluation_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_evaluation_result(self) -> "SetupEvaluation":
        if self.strategy_version != self.pattern.strategy_version:
            raise ValueError("evaluation and pattern strategy versions must match")
        if self.accepted:
            if self.setup is None or self.rejection_reasons:
                raise ValueError("accepted evaluation requires setup and no rejection reasons")
        elif self.setup is not None or not self.rejection_reasons:
            raise ValueError("rejected evaluation requires reasons and cannot contain setup")
        return self
