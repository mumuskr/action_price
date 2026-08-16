"""Descriptive and conditional statistics for completed trading setups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import time
from enum import StrEnum
from math import isfinite
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from brooks_trader.models import Direction, MarketRegime, PatternType, TradeRecord
from brooks_trader.models.common import DomainModel


class StatisticsScope(StrEnum):
    OVERALL = "OVERALL"
    PATTERN = "PATTERN"
    PATTERN_REGIME = "PATTERN_REGIME"
    PATTERN_SIGNAL_QUALITY = "PATTERN_SIGNAL_QUALITY"
    PATTERN_EMA_SLOPE = "PATTERN_EMA_SLOPE"
    PATTERN_SESSION = "PATTERN_SESSION"
    PATTERN_VOLATILITY = "PATTERN_VOLATILITY"


class SignalQualityBucket(StrEnum):
    WEAK = "WEAK"
    ACCEPTABLE = "ACCEPTABLE"
    STRONG = "STRONG"


class EmaSlopeBucket(StrEnum):
    OPPOSING = "OPPOSING"
    FLAT = "FLAT"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class SessionBucket(StrEnum):
    PREMARKET = "PREMARKET"
    OPEN = "OPEN"
    MIDDAY = "MIDDAY"
    CLOSE = "CLOSE"
    AFTER_HOURS = "AFTER_HOURS"


class VolatilityRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class SetupStatisticsConfig(BaseModel):
    """Configurable bucket boundaries for empirical setup statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_probability_sample: int = Field(ge=1)
    signal_body_weight: float = Field(ge=0)
    signal_close_weight: float = Field(ge=0)
    signal_quality_weak_max: float = Field(ge=0, le=1)
    signal_quality_strong_min: float = Field(ge=0, le=1)
    ema_slope_flat_max: float = Field(ge=0)
    ema_slope_strong_min: float = Field(ge=0)
    volatility_low_max: float = Field(ge=0)
    volatility_high_min: float = Field(ge=0)
    session_timezone: str = Field(min_length=1)
    regular_session_start: time
    open_session_end: time
    close_session_start: time
    regular_session_end: time

    @field_validator("session_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown session timezone: {value}") from error
        return value

    @model_validator(mode="after")
    def validate_boundaries(self) -> SetupStatisticsConfig:
        if self.signal_body_weight + self.signal_close_weight <= 0:
            raise ValueError("signal quality weights must have a positive total")
        if self.signal_quality_weak_max >= self.signal_quality_strong_min:
            raise ValueError("weak signal threshold must be below strong threshold")
        if self.ema_slope_flat_max >= self.ema_slope_strong_min:
            raise ValueError("flat EMA threshold must be below strong threshold")
        if self.volatility_low_max >= self.volatility_high_min:
            raise ValueError("low volatility threshold must be below high threshold")
        if not (
            self.regular_session_start
            < self.open_session_end
            <= self.close_session_start
            < self.regular_session_end
        ):
            raise ValueError("session boundaries must increase chronologically")
        return self


class SetupStatistics(DomainModel):
    """One empirical aggregate; pattern quality remains distinct from probability."""

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    scope: StatisticsScope
    pattern_type: PatternType | None = None
    market_regime: MarketRegime | None = None
    signal_bar_quality: SignalQualityBucket | None = None
    ema_slope_bucket: EmaSlopeBucket | None = None
    session: SessionBucket | None = None
    volatility_regime: VolatilityRegime | None = None
    total: int = Field(ge=1)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    breakevens: int = Field(ge=0)
    win_rate: float = Field(ge=0, le=1)
    probability_win: float | None = Field(default=None, ge=0, le=1)
    avg_win_r: float | None = None
    avg_loss_r: float | None = None
    avg_r: float
    expectancy_r: float
    profit_factor: float | None = Field(default=None, ge=0)
    max_drawdown: float = Field(ge=0)
    median_mfe: float = Field(ge=0)
    median_mae: float = Field(ge=0)
    median_mfe_r: float = Field(ge=0)
    median_mae_r: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts_and_values(self) -> SetupStatistics:
        if self.wins + self.losses + self.breakevens != self.total:
            raise ValueError("wins, losses, and breakevens must equal total")
        if abs(self.win_rate - self.wins / self.total) > 1e-12:
            raise ValueError("win_rate must equal wins / total")
        if abs(self.avg_r - self.expectancy_r) > 1e-12:
            raise ValueError("avg_r and expectancy_r must match")
        numeric = (
            self.avg_r,
            self.expectancy_r,
            self.max_drawdown,
            self.median_mfe,
            self.median_mae,
            self.median_mfe_r,
            self.median_mae_r,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("setup statistics must contain finite numeric values")
        return self


STATISTICS_COLUMNS = tuple(SetupStatistics.model_fields)


def load_setup_statistics_config(path: str | Path) -> SetupStatisticsConfig:
    """Load only Phase 7 bucket and sample-size configuration."""
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("statistics"), dict):
        raise ValueError("strategy configuration requires a statistics mapping")
    return SetupStatisticsConfig.model_validate(raw["statistics"])


def calculate_setup_statistics(
    trades: Sequence[TradeRecord],
    *,
    config: SetupStatisticsConfig,
) -> list[SetupStatistics]:
    """Calculate fixed, queryable aggregates without inventing missing samples."""
    if not trades:
        return []
    for trade in trades:
        if (
            trade.exit_time is None
            or trade.exit_price is None
            or trade.exit_reason is None
            or trade.pnl is None
            or trade.pnl_r is None
        ):
            raise ValueError("setup statistics require completed trades with PnL")
    enriched = [(trade, classify_trade_conditions(trade, config=config)) for trade in trades]
    grouped: dict[tuple[object, ...], list[TradeRecord]] = defaultdict(list)
    scopes: tuple[
        tuple[StatisticsScope, Callable[[TradeRecord, TradeConditions], tuple[object, ...]]],
        ...,
    ] = (
        (StatisticsScope.OVERALL, lambda trade, condition: ()),
        (StatisticsScope.PATTERN, lambda trade, condition: (trade.pattern_type,)),
        (
            StatisticsScope.PATTERN_REGIME,
            lambda trade, condition: (trade.pattern_type, trade.market_regime),
        ),
        (
            StatisticsScope.PATTERN_SIGNAL_QUALITY,
            lambda trade, condition: (trade.pattern_type, condition.signal_bar_quality),
        ),
        (
            StatisticsScope.PATTERN_EMA_SLOPE,
            lambda trade, condition: (trade.pattern_type, condition.ema_slope_bucket),
        ),
        (
            StatisticsScope.PATTERN_SESSION,
            lambda trade, condition: (trade.pattern_type, condition.session),
        ),
        (
            StatisticsScope.PATTERN_VOLATILITY,
            lambda trade, condition: (trade.pattern_type, condition.volatility_regime),
        ),
    )
    for trade, conditions in enriched:
        base = (trade.symbol, trade.timeframe, trade.strategy_version)
        for scope, key_builder in scopes:
            grouped[(*base, scope, *key_builder(trade, conditions))].append(trade)

    results: list[SetupStatistics] = []
    for key, group in grouped.items():
        symbol, timeframe, strategy_version, scope, *dimensions = key
        values = _scope_dimensions(scope, dimensions)
        results.append(
            _aggregate(
                group,
                symbol=str(symbol),
                timeframe=str(timeframe),
                strategy_version=str(strategy_version),
                scope=StatisticsScope(scope),
                config=config,
                **values,
            )
        )
    return sorted(results, key=_statistics_sort_key)


class TradeConditions(BaseModel):
    """Point-in-time buckets derived only from fields captured with the trade."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_bar_quality: SignalQualityBucket
    ema_slope_bucket: EmaSlopeBucket
    session: SessionBucket
    volatility_regime: VolatilityRegime


def classify_trade_conditions(
    trade: TradeRecord,
    *,
    config: SetupStatisticsConfig,
) -> TradeConditions:
    """Map stored setup inputs into configured descriptive buckets."""
    directional_close = (
        trade.signal_close_location
        if trade.direction == Direction.LONG
        else 1.0 - trade.signal_close_location
    )
    weight_total = config.signal_body_weight + config.signal_close_weight
    signal_quality = (
        config.signal_body_weight * trade.signal_body_ratio
        + config.signal_close_weight * directional_close
    ) / weight_total
    if signal_quality <= config.signal_quality_weak_max:
        signal_bucket = SignalQualityBucket.WEAK
    elif signal_quality >= config.signal_quality_strong_min:
        signal_bucket = SignalQualityBucket.STRONG
    else:
        signal_bucket = SignalQualityBucket.ACCEPTABLE

    directional_slope = (
        trade.ema_slope_ratio if trade.direction == Direction.LONG else -trade.ema_slope_ratio
    )
    if directional_slope < 0:
        ema_bucket = EmaSlopeBucket.OPPOSING
    elif directional_slope <= config.ema_slope_flat_max:
        ema_bucket = EmaSlopeBucket.FLAT
    elif directional_slope >= config.ema_slope_strong_min:
        ema_bucket = EmaSlopeBucket.STRONG
    else:
        ema_bucket = EmaSlopeBucket.MODERATE

    if trade.volatility_range_ratio <= config.volatility_low_max:
        volatility = VolatilityRegime.LOW
    elif trade.volatility_range_ratio >= config.volatility_high_min:
        volatility = VolatilityRegime.HIGH
    else:
        volatility = VolatilityRegime.NORMAL

    local_time = trade.entry_time.astimezone(ZoneInfo(config.session_timezone)).time()
    if local_time < config.regular_session_start:
        session = SessionBucket.PREMARKET
    elif local_time < config.open_session_end:
        session = SessionBucket.OPEN
    elif local_time < config.close_session_start:
        session = SessionBucket.MIDDAY
    elif local_time < config.regular_session_end:
        session = SessionBucket.CLOSE
    else:
        session = SessionBucket.AFTER_HOURS
    return TradeConditions(
        signal_bar_quality=signal_bucket,
        ema_slope_bucket=ema_bucket,
        session=session,
        volatility_regime=volatility,
    )


def statistics_to_frame(statistics: Sequence[SetupStatistics]) -> pd.DataFrame:
    """Convert statistics to a stable Parquet schema."""
    records = [statistic.model_dump(mode="json") for statistic in statistics]
    frame = pd.DataFrame.from_records(records, columns=STATISTICS_COLUMNS)
    string_columns = (
        "symbol",
        "timeframe",
        "strategy_version",
        "scope",
        "pattern_type",
        "market_regime",
        "signal_bar_quality",
        "ema_slope_bucket",
        "session",
        "volatility_regime",
    )
    integer_columns = ("total", "wins", "losses", "breakevens")
    float_columns = tuple(
        column for column in STATISTICS_COLUMNS if column not in {*string_columns, *integer_columns}
    )
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in integer_columns:
        frame[column] = frame[column].astype("Int64")
    for column in float_columns:
        frame[column] = frame[column].astype("Float64")
    return frame


def write_setup_statistics(
    statistics: Sequence[SetupStatistics],
    path: str | Path,
) -> Path:
    """Atomically persist empirical setup statistics as Parquet."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    statistics_to_frame(statistics).to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def _scope_dimensions(
    scope: object,
    dimensions: list[object],
) -> dict[str, object]:
    if scope == StatisticsScope.OVERALL:
        return {}
    pattern = dimensions[0]
    values: dict[str, object] = {"pattern_type": pattern}
    field_by_scope = {
        StatisticsScope.PATTERN_REGIME: "market_regime",
        StatisticsScope.PATTERN_SIGNAL_QUALITY: "signal_bar_quality",
        StatisticsScope.PATTERN_EMA_SLOPE: "ema_slope_bucket",
        StatisticsScope.PATTERN_SESSION: "session",
        StatisticsScope.PATTERN_VOLATILITY: "volatility_regime",
    }
    field = field_by_scope.get(StatisticsScope(scope))
    if field is not None:
        values[field] = dimensions[1]
    return values


def _aggregate(
    trades: Sequence[TradeRecord],
    *,
    symbol: str,
    timeframe: str,
    strategy_version: str,
    scope: StatisticsScope,
    config: SetupStatisticsConfig,
    pattern_type: PatternType | None = None,
    market_regime: MarketRegime | None = None,
    signal_bar_quality: SignalQualityBucket | None = None,
    ema_slope_bucket: EmaSlopeBucket | None = None,
    session: SessionBucket | None = None,
    volatility_regime: VolatilityRegime | None = None,
) -> SetupStatistics:
    ordered = sorted(
        trades, key=lambda trade: (trade.exit_time or trade.entry_time, trade.trade_id)
    )
    pnls = [float(trade.pnl or 0.0) for trade in ordered]
    pnl_rs = [float(trade.pnl_r or 0.0) for trade in ordered]
    wins = [value for value in pnl_rs if value > 0]
    losses = [value for value in pnl_rs if value < 0]
    total = len(ordered)
    probability = len(wins) / total if total >= config.minimum_probability_sample else None
    return SetupStatistics(
        symbol=symbol,
        timeframe=timeframe,
        strategy_version=strategy_version,
        scope=scope,
        pattern_type=pattern_type,
        market_regime=market_regime,
        signal_bar_quality=signal_bar_quality,
        ema_slope_bucket=ema_slope_bucket,
        session=session,
        volatility_regime=volatility_regime,
        total=total,
        wins=len(wins),
        losses=len(losses),
        breakevens=total - len(wins) - len(losses),
        win_rate=len(wins) / total,
        probability_win=probability,
        avg_win_r=sum(wins) / len(wins) if wins else None,
        avg_loss_r=sum(losses) / len(losses) if losses else None,
        avg_r=sum(pnl_rs) / total,
        expectancy_r=sum(pnl_rs) / total,
        profit_factor=_profit_factor(pnls),
        max_drawdown=_max_drawdown(pnls),
        median_mfe=median(trade.mfe for trade in ordered),
        median_mae=median(trade.mae for trade in ordered),
        median_mfe_r=median(trade.mfe_r for trade in ordered),
        median_mae_r=median(trade.mae_r for trade in ordered),
    )


def _profit_factor(pnls: Sequence[float]) -> float | None:
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    if gross_loss > 0:
        return gross_profit / gross_loss
    return None


def _max_drawdown(pnls: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    maximum = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _statistics_sort_key(statistic: SetupStatistics) -> tuple[str, ...]:
    return tuple(
        "" if value is None else str(value)
        for value in (
            statistic.symbol,
            statistic.timeframe,
            statistic.strategy_version,
            statistic.scope,
            statistic.pattern_type,
            statistic.market_regime,
            statistic.signal_bar_quality,
            statistic.ema_slope_bucket,
            statistic.session,
            statistic.volatility_regime,
        )
    )
