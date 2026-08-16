"""Second Entry With Trend setup evaluation, separate from trade execution."""

from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from brooks_trader.models import (
    Bar,
    Direction,
    MarketRegime,
    MarketState,
    PatternEvent,
    PatternType,
    SetupEvaluation,
    SetupType,
    TradeSetup,
)
from brooks_trader.strategy.catalog import StrategyModuleSelection


class SetupConfig(BaseModel):
    """Validated computational proxies for Second Entry With Trend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_pattern_score: float = Field(ge=0, le=1)
    minimum_context_score: float = Field(ge=0, le=1)
    minimum_signal_body_ratio: float = Field(ge=0, le=1)
    minimum_signal_close_location: float = Field(ge=0.5, le=1)
    minimum_directional_pressure: float = Field(ge=-1, le=1)
    maximum_pullback_depth_ranges: float = Field(gt=0)
    context_lookback: int = Field(ge=2)
    tight_range_lookback: int = Field(ge=2)
    tight_range_min_overlap: float = Field(ge=0, le=1)
    tight_range_max_span_ranges: float = Field(gt=0)
    climax_lookback: int = Field(ge=2)
    climax_min_trend_bars: int = Field(ge=2)
    minimum_room_to_target_r: float = Field(gt=0)
    reject_tight_trading_range: bool
    reject_recent_climax: bool
    reject_insufficient_room: bool

    @model_validator(mode="after")
    def validate_climax_window(self) -> "SetupConfig":
        if self.climax_min_trend_bars > self.climax_lookback:
            raise ValueError("climax_min_trend_bars cannot exceed climax_lookback")
        return self


class EntryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    second_entries_only: bool
    order_type: Literal["stop"]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_mode: Literal["structural"]


class ExitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["1R", "2R"]
    reward_multiple: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_reward_multiple(self) -> "ExitConfig":
        expected = 1.0 if self.mode == "1R" else 2.0
        if self.reward_multiple != expected:
            raise ValueError("reward_multiple must match the configured 1R or 2R exit mode")
        return self


class MarketExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1)
    tick_size: float = Field(gt=0)
    point_value: float = Field(gt=0)


class SetupEngineConfig(BaseModel):
    """All configuration required to evaluate a setup for one market."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    setup: SetupConfig
    entry: EntryConfig
    risk: RiskConfig
    exit: ExitConfig
    market: MarketExecutionConfig
    modules: StrategyModuleSelection = Field(default_factory=StrategyModuleSelection)


def load_setup_engine_config(
    strategy_path: str | Path,
    markets_path: str | Path,
    *,
    symbol: str,
    module_overrides: Mapping[str, Any] | StrategyModuleSelection | None = None,
) -> tuple[SetupEngineConfig, str]:
    """Load Phase 5 settings and one market's tick size."""
    strategy_raw = _read_mapping(strategy_path)
    markets_raw = _read_mapping(markets_path)
    strategy = _require_mapping(strategy_raw, "strategy")
    markets = _require_mapping(markets_raw, "markets")
    normalized_symbol = symbol.strip().upper()
    market = markets.get(normalized_symbol)
    if not isinstance(market, Mapping):
        raise ValueError(f"market {normalized_symbol!r} is not configured")
    version = strategy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("strategy.version must be a non-empty string")
    raw_modules = strategy_raw.get("modules")
    if not isinstance(raw_modules, Mapping):
        raw_modules = {}
    override_values = (
        module_overrides.model_dump()
        if isinstance(module_overrides, StrategyModuleSelection)
        else module_overrides
    )
    modules = StrategyModuleSelection.from_values(raw_modules, overrides=override_values)
    return (
        SetupEngineConfig.model_validate(
            {
                "setup": _require_mapping(strategy_raw, "setup"),
                "entry": _require_mapping(strategy_raw, "entry"),
                "risk": {"stop_mode": _require_mapping(strategy_raw, "risk").get("stop_mode")},
                "exit": _require_mapping(strategy_raw, "exit"),
                "market": {
                    "symbol": normalized_symbol,
                    "tick_size": market.get("tick_size"),
                    "point_value": market.get("point_value"),
                },
                "modules": modules,
            }
        ),
        version,
    )


class SetupEngine:
    """Evaluate H2/L2 patterns as setups without creating an order."""

    def __init__(self, config: SetupEngineConfig, *, strategy_version: str) -> None:
        if not strategy_version.strip():
            raise ValueError("strategy_version cannot be empty")
        self.config = config
        self.strategy_version = strategy_version

    def evaluate(
        self,
        pattern: PatternEvent,
        bars: Sequence[Bar],
        features: pd.DataFrame,
        contexts: Sequence[MarketState],
    ) -> SetupEvaluation:
        """Evaluate one point-in-time pattern using data no later than its signal bar."""
        signal_index = pattern.signal_index
        self._validate_inputs(pattern, bars, features, contexts)
        known_bars = bars[: signal_index + 1]
        known_features = features.iloc[: signal_index + 1].reset_index(drop=True)
        known_contexts = contexts[: signal_index + 1]
        signal_bar = known_bars[-1]
        signal_feature = known_features.iloc[-1]
        context = known_contexts[-1]
        reasons: list[str] = []
        warnings: list[str] = []
        rejections: list[str] = []

        direction = pattern.direction
        expected_pattern = PatternType.H2 if direction == Direction.LONG else PatternType.L2
        expected_setup = (
            SetupType.H2_WITH_TREND if direction == Direction.LONG else SetupType.L2_WITH_TREND
        )
        direction_module_enabled = (
            self.config.modules.h2_with_trend
            if direction == Direction.LONG
            else self.config.modules.l2_with_trend
        )
        if not direction_module_enabled:
            rejections.append("strategy_module_disabled")
        if pattern.pattern_type != expected_pattern:
            rejections.append("second_entry_pattern_required")
        if self.config.modules.market_regime_filter and not _regime_with_direction(
            context, direction
        ):
            rejections.append("market_regime_not_with_trend")
        if self.config.modules.ema_alignment_filter and not _ema_with_direction(
            signal_feature, direction
        ):
            rejections.append("ema_not_aligned")
        if (
            self.config.modules.pattern_quality_filter
            and pattern.confidence_score < self.config.setup.minimum_pattern_score
        ):
            rejections.append("pattern_quality_below_minimum")
        raw_context_score = _directional_value(context.trend_score, direction)
        context_score = max(0.0, min(1.0, raw_context_score))
        if (
            self.config.modules.context_quality_filter
            and raw_context_score < self.config.setup.minimum_context_score
        ):
            rejections.append("context_score_below_minimum")

        if self.config.modules.signal_bar_filter:
            if float(signal_feature["body_ratio"]) < self.config.setup.minimum_signal_body_ratio:
                rejections.append("signal_bar_body_too_small")
            if not _signal_close_is_acceptable(signal_feature, direction, self.config.setup):
                rejections.append("signal_bar_close_too_weak")
        if not _directional_bar(signal_feature, direction):
            warnings.append("signal_bar_not_directional")
        if (
            self.config.modules.pressure_filter
            and _directional_value(context.pressure_score, direction)
            < self.config.setup.minimum_directional_pressure
        ):
            rejections.append("opposite_or_insufficient_pressure")

        pullback = known_bars[pattern.start_index : signal_index + 1]
        pullback_depth, average_range = _pullback_depth_in_ranges(pullback)
        if (
            self.config.modules.pullback_depth_filter
            and pullback_depth > self.config.setup.maximum_pullback_depth_ranges
        ):
            rejections.append("pullback_too_deep")

        tight_range = _is_tight_trading_range(known_bars, known_features, self.config.setup)
        if tight_range and self.config.modules.tight_trading_range_filter:
            if self.config.setup.reject_tight_trading_range:
                rejections.append("tight_trading_range")
            else:
                warnings.append("tight_trading_range")

        recent_climax = _has_recent_climax(known_features, direction, self.config.setup)
        if recent_climax and self.config.modules.recent_climax_filter:
            if self.config.setup.reject_recent_climax:
                rejections.append("recent_climax")
            else:
                warnings.append("recent_climax")

        entry = _stop_entry(signal_bar, direction, self.config.market.tick_size)
        initial_stop = _structural_stop(pullback, direction, self.config.market.tick_size)
        risk = abs(entry - initial_stop)
        if risk <= 0:
            rejections.append("non_positive_initial_risk")
        target = _target(
            entry,
            risk,
            direction,
            self.config.exit.reward_multiple,
            self.config.market.tick_size,
        )
        room_r = _room_to_target_r(
            known_bars,
            pattern.start_index,
            entry,
            risk,
            direction,
            self.config.setup.context_lookback,
        )
        if (
            self.config.modules.room_to_target_filter
            and room_r is not None
            and room_r < self.config.setup.minimum_room_to_target_r
        ):
            if self.config.setup.reject_insufficient_room:
                rejections.append("insufficient_room_to_target")
            else:
                warnings.append("insufficient_room_to_target")

        metadata = {
            "symbol": self.config.market.symbol,
            "tick_size": self.config.market.tick_size,
            "stop_mode": self.config.risk.stop_mode,
            "exit_mode": self.config.exit.mode,
            "reward_multiple": self.config.exit.reward_multiple,
            "pullback_depth_ranges": pullback_depth,
            "average_pullback_bar_range": average_range,
            "tight_trading_range": tight_range,
            "recent_climax": recent_climax,
            "room_to_target_r": room_r,
            "signal_body_ratio": float(signal_feature["body_ratio"]),
            "signal_close_location": float(signal_feature["close_location"]),
            "raw_context_score": raw_context_score,
            "ema_slope_ratio": _ema_slope_ratio(signal_feature),
            "volatility_range_ratio": _recent_average_range_ratio(
                known_bars,
                self.config.setup.context_lookback,
            ),
            "probability_win": None,
            "expected_value": None,
        }
        if rejections:
            return SetupEvaluation(
                evaluated_at=signal_bar.timestamp,
                pattern=pattern,
                accepted=False,
                rejection_reasons=list(dict.fromkeys(rejections)),
                warnings=warnings,
                metadata=metadata,
                strategy_version=self.strategy_version,
            )

        reasons.append("second_entry_pattern")
        if self.config.modules.market_regime_filter:
            reasons.append("market_regime_with_trend")
        if self.config.modules.ema_alignment_filter:
            reasons.append("ema_aligned")
        if (
            self.config.modules.pattern_quality_filter
            and self.config.modules.context_quality_filter
        ):
            reasons.append("context_and_pattern_quality_accepted")
        elif self.config.modules.pattern_quality_filter:
            reasons.append("pattern_quality_accepted")
        elif self.config.modules.context_quality_filter:
            reasons.append("context_quality_accepted")
        if self.config.modules.signal_bar_filter:
            reasons.append("signal_bar_quality_accepted")
        if self.config.modules.pressure_filter:
            reasons.append("directional_pressure_accepted")
        if self.config.modules.pullback_depth_filter:
            reasons.append("pullback_depth_accepted")
        if self.config.modules.tight_trading_range_filter:
            reasons.append("trading_range_filter_passed")
        if self.config.modules.recent_climax_filter:
            reasons.append("climax_filter_passed")
        if self.config.modules.room_to_target_filter:
            reasons.append("room_to_target_accepted")
        setup = TradeSetup(
            setup_type=expected_setup,
            direction=direction,
            evaluated_at=signal_bar.timestamp,
            signal_bar_index=signal_index,
            pattern=pattern,
            market_state=context,
            pattern_score=pattern.confidence_score,
            context_score=context_score,
            entry=entry,
            initial_stop=initial_stop,
            target=target,
            invalidation="price reaches initial_stop before target",
            reasons=reasons,
            warnings=warnings,
            metadata=metadata,
            strategy_version=self.strategy_version,
        )
        return SetupEvaluation(
            evaluated_at=signal_bar.timestamp,
            pattern=pattern,
            accepted=True,
            setup=setup,
            warnings=warnings,
            metadata=metadata,
            strategy_version=self.strategy_version,
        )

    def _validate_inputs(
        self,
        pattern: PatternEvent,
        bars: Sequence[Bar],
        features: pd.DataFrame,
        contexts: Sequence[MarketState],
    ) -> None:
        if pattern.strategy_version != self.strategy_version:
            raise ValueError("pattern and SetupEngine strategy versions must match")
        if len(bars) != len(features) or len(bars) != len(contexts):
            raise ValueError("bars, features, and contexts must have equal lengths")
        if pattern.signal_index >= len(bars):
            raise ValueError("pattern signal_index is outside supplied history")
        validation_start = max(
            0,
            min(
                pattern.start_index,
                pattern.signal_index - self.config.setup.context_lookback,
            ),
        )
        for position in range(validation_start, pattern.signal_index + 1):
            if int(features.iloc[position]["bar_index"]) != position:
                raise ValueError("feature bar_index must match its history position")
            if contexts[position].bar_index != position:
                raise ValueError("market-state bar_index must match its history position")
            if not (
                bars[position].timestamp
                == features.iloc[position]["timestamp"]
                == contexts[position].timestamp
            ):
                raise ValueError("bar, feature, and market-state timestamps must match")
        if pattern.signal_time != bars[pattern.signal_index].timestamp:
            raise ValueError("pattern signal time does not match supplied signal bar")
        if pattern.context != contexts[pattern.signal_index]:
            raise ValueError("pattern context does not match supplied market state")


def _regime_with_direction(context: MarketState, direction: Direction) -> bool:
    allowed = (
        {MarketRegime.BULL_TREND, MarketRegime.STRONG_BULL_TREND}
        if direction == Direction.LONG
        else {MarketRegime.BEAR_TREND, MarketRegime.STRONG_BEAR_TREND}
    )
    return context.regime in allowed


def _ema_with_direction(feature: pd.Series, direction: Direction) -> bool:
    slope = feature["ema_slope"]
    if pd.isna(slope):
        return False
    distance = float(feature["distance_to_ema"])
    return (
        (distance > 0 and float(slope) > 0)
        if direction == Direction.LONG
        else (distance < 0 and float(slope) < 0)
    )


def _ema_slope_ratio(feature: pd.Series) -> float:
    ema = float(feature["ema20"])
    slope = feature["ema_slope"]
    if pd.isna(slope) or ema == 0:
        return 0.0
    return float(slope) / abs(ema)


def _recent_average_range_ratio(bars: Sequence[Bar], lookback: int) -> float:
    window = bars[-lookback:]
    average_range = sum(bar.high - bar.low for bar in window) / len(window)
    reference_price = abs(window[-1].close)
    return average_range / reference_price if reference_price > 0 else 0.0


def _signal_close_is_acceptable(
    feature: pd.Series,
    direction: Direction,
    config: SetupConfig,
) -> bool:
    close_location = float(feature["close_location"])
    return (
        close_location >= config.minimum_signal_close_location
        if direction == Direction.LONG
        else close_location <= 1.0 - config.minimum_signal_close_location
    )


def _directional_bar(feature: pd.Series, direction: Direction) -> bool:
    return bool(feature["bull_bar"] if direction == Direction.LONG else feature["bear_bar"])


def _directional_value(value: float, direction: Direction) -> float:
    return value if direction == Direction.LONG else -value


def _pullback_depth_in_ranges(pullback: Sequence[Bar]) -> tuple[float, float]:
    ranges = [bar.high - bar.low for bar in pullback]
    average_range = sum(ranges) / len(ranges)
    span = max(bar.high for bar in pullback) - min(bar.low for bar in pullback)
    return (span / average_range if average_range > 0 else 0.0), average_range


def _is_tight_trading_range(
    bars: Sequence[Bar],
    features: pd.DataFrame,
    config: SetupConfig,
) -> bool:
    lookback = min(config.tight_range_lookback, len(bars))
    if lookback < config.tight_range_lookback:
        return False
    window_bars = bars[-lookback:]
    window_features = features.iloc[-lookback:]
    average_range = sum(bar.high - bar.low for bar in window_bars) / lookback
    if average_range <= 0:
        return True
    span = max(bar.high for bar in window_bars) - min(bar.low for bar in window_bars)
    overlap = float(window_features["overlap_previous"].iloc[1:].mean())
    return (
        overlap >= config.tight_range_min_overlap
        and span / average_range <= config.tight_range_max_span_ranges
    )


def _has_recent_climax(
    features: pd.DataFrame,
    direction: Direction,
    config: SetupConfig,
) -> bool:
    if len(features) < config.climax_lookback:
        return False
    window = features.iloc[-config.climax_lookback :]
    directional = window["bull_bar"] if direction == Direction.LONG else window["bear_bar"]
    return int((window["trend_bar"] & directional).sum()) >= config.climax_min_trend_bars


def _stop_entry(bar: Bar, direction: Direction, tick_size: float) -> float:
    raw = bar.high + tick_size if direction == Direction.LONG else bar.low - tick_size
    return _round_to_tick(raw, tick_size, up=direction == Direction.LONG)


def _structural_stop(
    pullback: Sequence[Bar],
    direction: Direction,
    tick_size: float,
) -> float:
    raw = (
        min(bar.low for bar in pullback) - tick_size
        if direction == Direction.LONG
        else max(bar.high for bar in pullback) + tick_size
    )
    return _round_to_tick(raw, tick_size, up=direction == Direction.SHORT)


def _target(
    entry: float,
    risk: float,
    direction: Direction,
    reward_multiple: float,
    tick_size: float,
) -> float:
    tick_decimal = Decimal(str(tick_size))
    entry_decimal = Decimal(str(entry))
    risk_ticks = (Decimal(str(risk)) / tick_decimal).to_integral_value(rounding=ROUND_HALF_UP)
    move = risk_ticks * tick_decimal * Decimal(str(reward_multiple))
    raw = entry_decimal + move if direction == Direction.LONG else entry_decimal - move
    rounding = ROUND_CEILING if direction == Direction.LONG else ROUND_FLOOR
    target_ticks = (raw / tick_decimal).to_integral_value(rounding=rounding)
    return float(target_ticks * tick_decimal)


def _room_to_target_r(
    bars: Sequence[Bar],
    start_index: int,
    entry: float,
    risk: float,
    direction: Direction,
    lookback: int,
) -> float | None:
    if risk <= 0 or start_index <= 0:
        return None
    prior = bars[max(0, start_index - lookback) : start_index]
    if not prior:
        return None
    if direction == Direction.LONG:
        resistance_levels = [bar.high for bar in prior if bar.high > entry]
        if not resistance_levels:
            return None
        return (min(resistance_levels) - entry) / risk
    support_levels = [bar.low for bar in prior if bar.low < entry]
    if not support_levels:
        return None
    return (entry - max(support_levels)) / risk


def _round_to_tick(price: float, tick_size: float, *, up: bool) -> float:
    price_decimal = Decimal(str(price))
    tick_decimal = Decimal(str(tick_size))
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    ticks = (price_decimal / tick_decimal).to_integral_value(rounding=rounding)
    return float(ticks * tick_decimal)


def _read_mapping(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration must be a mapping: {source}")
    return raw


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {key!r} must be a mapping")
    return value
