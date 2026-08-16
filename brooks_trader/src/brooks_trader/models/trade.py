"""Execution intent, order, and auditable trade-record contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import Direction, DomainModel
from brooks_trader.models.market_state import MarketRegime, MarketState
from brooks_trader.models.pattern import PatternType
from brooks_trader.models.setup import TradeSetup


class OrderType(StrEnum):
    MARKET = "MARKET"
    STOP = "STOP"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeIntent(DomainModel):
    """A strategy decision that must pass risk and human-confirmation gates."""

    id: str = Field(min_length=1)
    created_at: datetime
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    direction: Direction
    quantity: float = Field(gt=0)
    order_type: OrderType
    entry_price: float | None = Field(default=None, gt=0)
    stop_price: float = Field(gt=0)
    target_price: float | None = Field(default=None, gt=0)
    setup: TradeSetup
    strategy_version: str = Field(min_length=1)
    human_confirmation_required: bool = True

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_entry_price(self) -> "TradeIntent":
        if self.order_type is not OrderType.MARKET and self.entry_price is None:
            raise ValueError("STOP and LIMIT trade intents require entry_price")
        return self


class Order(DomainModel):
    """Broker-facing order state."""

    id: str = Field(min_length=1)
    timestamp: datetime
    direction: Direction
    order_type: OrderType
    quantity: float = Field(gt=0)
    price: float | None = Field(default=None, gt=0)
    status: OrderStatus = OrderStatus.PENDING
    filled_at: datetime | None = None
    filled_price: float | None = Field(default=None, gt=0)
    rejection_reason: str | None = None

    @field_validator("timestamp", "filled_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("order timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_order_state(self) -> "Order":
        if self.order_type is not OrderType.MARKET and self.price is None:
            raise ValueError("STOP and LIMIT orders require price")
        if self.status is OrderStatus.FILLED and (
            self.filled_at is None or self.filled_price is None
        ):
            raise ValueError("filled orders require filled_at and filled_price")
        return self


class TradeRecord(DomainModel):
    """A reproducible trade log record with point-in-time context snapshots."""

    trade_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    setup: str = Field(min_length=1)
    pattern_type: PatternType
    direction: Direction
    market_regime: MarketRegime
    pattern_score: float = Field(ge=0, le=1)
    context_score: float = Field(ge=0, le=1)
    signal_body_ratio: float = Field(ge=0, le=1)
    signal_close_location: float = Field(ge=0, le=1)
    ema_slope_ratio: float
    volatility_range_ratio: float = Field(ge=0)
    entry_time: datetime
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    point_value: float = Field(gt=0)
    slippage_ticks: int = Field(ge=0)
    stop_price: float = Field(gt=0)
    target_price: float | None = Field(default=None, gt=0)
    exit_time: datetime | None = None
    exit_price: float | None = Field(default=None, gt=0)
    initial_risk: float = Field(gt=0)
    gross_pnl: float | None = None
    pnl: float | None = None
    commission: float = Field(default=0, ge=0)
    pnl_r: float | None = None
    mfe: float = Field(default=0, ge=0)
    mae: float = Field(default=0, ge=0)
    mfe_r: float = Field(default=0, ge=0)
    mae_r: float = Field(default=0, ge=0)
    bars_held: int = Field(default=0, ge=0)
    signal_bar_index: int = Field(ge=0)
    market_state: MarketState
    pattern_metadata: dict[str, Any] = Field(default_factory=dict)
    entry_reason: str = Field(min_length=1)
    exit_reason: str | None = None
    strategy_version: str = Field(min_length=1)

    @field_validator("entry_time", "exit_time")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trade timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_exit_fields(self) -> "TradeRecord":
        exit_fields = (self.exit_time, self.exit_price, self.exit_reason)
        if any(value is not None for value in exit_fields) and not all(
            value is not None for value in exit_fields
        ):
            raise ValueError("exit_time, exit_price, and exit_reason must be set together")
        if self.exit_time is not None and self.exit_time < self.entry_time:
            raise ValueError("exit_time cannot precede entry_time")
        return self
