"""Transparent, incremental market-context component score proxies."""

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, tanh
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentWeights(BaseModel):
    """Weights for the explanatory component score proxy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ema: float = Field(ge=0)
    structure: float = Field(ge=0)
    pressure: float = Field(ge=0)
    overlap: float = Field(ge=0)
    breakout: float = Field(ge=0)

    @model_validator(mode="after")
    def require_positive_total(self) -> "ComponentWeights":
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one component weight must be positive")
        return self


class MarketContextConfig(BaseModel):
    """Validated Phase 3 thresholds and lookbacks.

    These values are computational choices for research. They are not presented as
    explicit formulas from Brooks' books.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ema_distance_scale: float = Field(gt=0)
    ema_slope_scale: float = Field(gt=0)
    ema_distance_weight: float = Field(ge=0)
    ema_slope_weight: float = Field(ge=0)
    structure_lookback: int = Field(ge=1)
    pressure_lookback: int = Field(ge=1)
    overlap_lookback: int = Field(ge=1)
    breakout_lookback: int = Field(ge=1)
    strong_bull_threshold: float = Field(ge=-1, le=1)
    bull_threshold: float = Field(ge=-1, le=1)
    bear_threshold: float = Field(ge=-1, le=1)
    trading_range_threshold: float = Field(ge=0, le=1)
    strong_bear_threshold: float = Field(ge=-1, le=1)
    always_in_score_threshold: float = Field(ge=0, le=1)
    always_in_confirmation_bars: int = Field(ge=1)
    component_weights: ComponentWeights

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "MarketContextConfig":
        if not (
            self.strong_bear_threshold
            < self.bear_threshold
            < -self.trading_range_threshold
            <= self.trading_range_threshold
            < self.bull_threshold
            < self.strong_bull_threshold
        ):
            raise ValueError("regime thresholds must bound the configured trading-range threshold")
        if self.ema_distance_weight + self.ema_slope_weight <= 0:
            raise ValueError("EMA distance/slope weights must have a positive total")
        return self


@dataclass(frozen=True)
class ContextScores:
    """One point-in-time set of bounded explanatory scores."""

    ema_score: float
    structure_score: float
    pressure_score: float
    overlap_score: float
    breakout_score: float
    trend_score: float


class TrendScoreTracker:
    """Incrementally calculate context scores from one closed BarFeatures row."""

    def __init__(self, config: MarketContextConfig) -> None:
        self.config = config
        self._structure_events: deque[float] = deque(maxlen=config.structure_lookback)
        self._pressure_events: deque[float] = deque(maxlen=config.pressure_lookback)
        self._overlap_events: deque[float] = deque(maxlen=config.overlap_lookback)
        self._breakout_events: deque[float] = deque(maxlen=config.breakout_lookback)
        self._previous_breakout_direction = 0.0

    def update(self, feature: Mapping[str, Any]) -> ContextScores:
        """Consume one feature row and return scores available at that same bar."""
        ema_score = self._ema_score(feature)

        structure_event = (
            float(_boolean(feature, "higher_high"))
            + float(_boolean(feature, "higher_low"))
            - float(_boolean(feature, "lower_high"))
            - float(_boolean(feature, "lower_low"))
        ) / 2.0
        self._structure_events.append(structure_event)
        structure_score = _mean(self._structure_events)

        body_ratio = _number(feature, "body_ratio")
        close_location = _number(feature, "close_location")
        bullish_pressure = body_ratio * close_location if _boolean(feature, "bull_bar") else 0.0
        bearish_pressure = (
            body_ratio * (1.0 - close_location) if _boolean(feature, "bear_bar") else 0.0
        )
        self._pressure_events.append(bullish_pressure - bearish_pressure)
        pressure_score = _mean(self._pressure_events)

        overlap_event = (1.0 - _number(feature, "overlap_previous")) * _sign(ema_score)
        self._overlap_events.append(overlap_event)
        overlap_score = _mean(self._overlap_events)

        direction = float(_boolean(feature, "bull_bar")) - float(_boolean(feature, "bear_bar"))
        breakout_direction = direction if _boolean(feature, "trend_bar") else 0.0
        follow_through = (
            direction if direction != 0 and direction == self._previous_breakout_direction else 0.0
        )
        self._breakout_events.append(0.5 * breakout_direction + 0.5 * follow_through)
        breakout_score = _mean(self._breakout_events)
        self._previous_breakout_direction = breakout_direction

        components = {
            "ema": ema_score,
            "structure": structure_score,
            "pressure": pressure_score,
            "overlap": overlap_score,
            "breakout": breakout_score,
        }
        weights = self.config.component_weights.model_dump()
        weight_total = sum(weights.values())
        trend_score = sum(weights[name] * components[name] for name in components) / weight_total
        return ContextScores(
            ema_score=_bounded(ema_score),
            structure_score=_bounded(structure_score),
            pressure_score=_bounded(pressure_score),
            overlap_score=_bounded(overlap_score),
            breakout_score=_bounded(breakout_score),
            trend_score=_bounded(trend_score),
        )

    def _ema_score(self, feature: Mapping[str, Any]) -> float:
        ema = _number(feature, "ema20")
        denominator = abs(ema)
        if denominator == 0:
            return 0.0
        distance_ratio = _number(feature, "distance_to_ema") / denominator
        slope = _optional_number(feature, "ema_slope")
        slope_ratio = 0.0 if slope is None else slope / denominator
        distance_component = tanh(distance_ratio / self.config.ema_distance_scale)
        slope_component = tanh(slope_ratio / self.config.ema_slope_scale)
        weight_total = self.config.ema_distance_weight + self.config.ema_slope_weight
        return (
            self.config.ema_distance_weight * distance_component
            + self.config.ema_slope_weight * slope_component
        ) / weight_total


def load_market_context_config(path: str | Path) -> tuple[MarketContextConfig, str]:
    """Load Phase 3 trend settings and the strategy version from YAML."""
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"strategy configuration must be a mapping: {source}")
    trend = _require_mapping(raw, "trend")
    strategy = _require_mapping(raw, "strategy")
    config_values = {key: trend.get(key) for key in MarketContextConfig.model_fields}
    config = MarketContextConfig.model_validate(config_values)
    version = strategy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("strategy.version must be a non-empty string")
    return config, version


def calculate_context_scores(
    features: pd.DataFrame,
    *,
    config: MarketContextConfig,
) -> pd.DataFrame:
    """Calculate scores through the same incremental path used by real-time updates."""
    tracker = TrendScoreTracker(config)
    records = [asdict(tracker.update(row)) for _, row in features.iterrows()]
    return pd.DataFrame.from_records(records, columns=ContextScores.__annotations__)


def feature_mapping(feature: Any) -> Mapping[str, Any]:
    """Convert a model or mapping into the tracker input contract."""
    if isinstance(feature, pd.Series):
        return feature.to_dict()
    if isinstance(feature, Mapping):
        return feature
    model_dump = getattr(feature, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        if isinstance(value, Mapping):
            return value
    raise TypeError("feature must be a mapping or Pydantic model")


def _bounded(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _mean(values: deque[float]) -> float:
    return sum(values) / len(values)


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _number(feature: Mapping[str, Any], key: str) -> float:
    if key not in feature:
        raise ValueError(f"feature row is missing {key!r}")
    value = float(feature[key])
    if not isfinite(value):
        raise ValueError(f"feature {key!r} must be finite")
    return value


def _optional_number(feature: Mapping[str, Any], key: str) -> float | None:
    if key not in feature or pd.isna(feature[key]):
        return None
    return _number(feature, key)


def _boolean(feature: Mapping[str, Any], key: str) -> bool:
    if key not in feature:
        raise ValueError(f"feature row is missing {key!r}")
    return bool(feature[key])


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"strategy configuration section {key!r} must be a mapping")
    return value
