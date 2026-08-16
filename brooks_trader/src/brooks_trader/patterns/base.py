"""Shared contracts and validation for causal pattern state machines."""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from brooks_trader.models import (
    AlwaysInState,
    Bar,
    Direction,
    MarketRegime,
    MarketState,
    PatternEvent,
    PatternType,
)
from brooks_trader.models.common import DomainModel


class PatternQualityWeights(BaseModel):
    """Weights for pattern quality, which is not a probability estimate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: float = Field(ge=0)
    signal_close: float = Field(ge=0)
    directional_body: float = Field(ge=0)

    @model_validator(mode="after")
    def require_positive_total(self) -> "PatternQualityWeights":
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one pattern quality weight must be positive")
        return self


class PatternDetectorConfig(BaseModel):
    """Validated Phase 4 pattern settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pullback_min_bars: int = Field(ge=1)
    pullback_max_bars: int = Field(ge=1)
    debug_transitions: bool
    quality_weights: PatternQualityWeights

    @model_validator(mode="after")
    def validate_pullback_limits(self) -> "PatternDetectorConfig":
        if self.pullback_max_bars < self.pullback_min_bars:
            raise ValueError("pullback_max_bars cannot be below pullback_min_bars")
        return self


class DetectorTransition(DomainModel):
    """A traceable state transition for detector debugging."""

    detector: str = Field(min_length=1)
    timestamp: datetime
    bar_index: int = Field(ge=0)
    state_before: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    state_after: str = Field(min_length=1)
    pattern_type: PatternType | None = None


class DetectorState(StrEnum):
    """Marker base for explicit detector state enums."""


class StatefulPatternDetector:
    """Common input validation, event creation, and transition logging."""

    detector_name = "pattern"

    def __init__(self, config: PatternDetectorConfig, *, strategy_version: str) -> None:
        if not strategy_version.strip():
            raise ValueError("strategy_version cannot be empty")
        self.config = config
        self.strategy_version = strategy_version
        self.debug_log: list[DetectorTransition] = []
        self._previous_bar: Bar | None = None
        self._last_bar_index: int | None = None
        self._last_timestamp: datetime | None = None

    def reset_base(self) -> None:
        """Reset common stream state and transition history."""
        self.debug_log.clear()
        self._previous_bar = None
        self._last_bar_index = None
        self._last_timestamp = None

    def begin_update(
        self,
        bar: Bar,
        feature: Any,
        context: MarketState,
    ) -> tuple[Mapping[str, Any], Bar | None]:
        """Validate one synchronized Bar/BarFeatures/MarketState observation."""
        row = feature_mapping(feature)
        bar_index = int(row["bar_index"])
        timestamp = row["timestamp"]
        if timestamp != bar.timestamp or context.timestamp != bar.timestamp:
            raise ValueError("bar, feature, and market-state timestamps must match")
        if context.bar_index != bar_index:
            raise ValueError("feature and market-state bar_index values must match")
        if context.strategy_version != self.strategy_version:
            raise ValueError("market-state and detector strategy versions must match")
        if self._last_bar_index is not None and bar_index <= self._last_bar_index:
            raise ValueError("pattern detector bar_index must increase strictly")
        if self._last_timestamp is not None and bar.timestamp <= self._last_timestamp:
            raise ValueError("pattern detector timestamps must increase strictly")

        previous = self._previous_bar
        self._previous_bar = bar
        self._last_bar_index = bar_index
        self._last_timestamp = bar.timestamp
        return row, previous

    def record_transition(
        self,
        *,
        bar: Bar,
        bar_index: int,
        state_before: StrEnum,
        condition: str,
        state_after: StrEnum,
        pattern_type: PatternType | None = None,
    ) -> None:
        """Append a state change when transition debugging is enabled."""
        if not self.config.debug_transitions:
            return
        self.debug_log.append(
            DetectorTransition(
                detector=self.detector_name,
                timestamp=bar.timestamp,
                bar_index=bar_index,
                state_before=state_before.value,
                condition=condition,
                state_after=state_after.value,
                pattern_type=pattern_type,
            )
        )

    def build_event(
        self,
        *,
        pattern_type: PatternType,
        direction: Direction,
        bar: Bar,
        feature: Mapping[str, Any],
        context: MarketState,
        start_index: int,
        pullback_bars: int,
        attempt_number: int,
        state_before: StrEnum,
    ) -> PatternEvent:
        """Build a PatternEvent with a transparent non-probability quality score."""
        bar_index = int(feature["bar_index"])
        trigger_price = bar.high if direction == Direction.LONG else bar.low
        return PatternEvent(
            pattern_type=pattern_type,
            direction=direction,
            start_index=start_index,
            signal_index=bar_index,
            signal_time=bar.timestamp,
            trigger_price=trigger_price,
            context=context,
            confidence_score=pattern_quality_score(
                feature,
                context=context,
                direction=direction,
                weights=self.config.quality_weights,
            ),
            metadata={
                "attempt_number": attempt_number,
                "pullback_bars": pullback_bars,
                "state_before": state_before.value,
                "quality_score_is_probability": False,
                "computational_proxy": "first_or_second_prior_bar_break_attempt",
            },
            strategy_version=self.strategy_version,
        )


def load_pattern_detector_config(path: str | Path) -> tuple[PatternDetectorConfig, str]:
    """Load Phase 4 pattern settings and strategy version from YAML."""
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"strategy configuration must be a mapping: {source}")
    patterns = _require_mapping(raw, "patterns")
    strategy = _require_mapping(raw, "strategy")
    config = PatternDetectorConfig.model_validate(patterns)
    version = strategy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("strategy.version must be a non-empty string")
    return config, version


def is_long_pattern_context(context: MarketState) -> bool:
    """Accept an explicit bull regime or a non-bearish Always In Long context."""
    if context.regime in {MarketRegime.BULL_TREND, MarketRegime.STRONG_BULL_TREND}:
        return True
    return context.always_in == AlwaysInState.ALWAYS_IN_LONG and context.regime not in {
        MarketRegime.BEAR_TREND,
        MarketRegime.STRONG_BEAR_TREND,
    }


def is_short_pattern_context(context: MarketState) -> bool:
    """Accept an explicit bear regime or a non-bullish Always In Short context."""
    if context.regime in {MarketRegime.BEAR_TREND, MarketRegime.STRONG_BEAR_TREND}:
        return True
    return context.always_in == AlwaysInState.ALWAYS_IN_SHORT and context.regime not in {
        MarketRegime.BULL_TREND,
        MarketRegime.STRONG_BULL_TREND,
    }


def pattern_quality_score(
    feature: Mapping[str, Any],
    *,
    context: MarketState,
    direction: Direction,
    weights: PatternQualityWeights,
) -> float:
    """Return pattern quality, explicitly not historical win probability."""
    context_quality = (
        max(0.0, context.trend_score)
        if direction == Direction.LONG
        else max(0.0, -context.trend_score)
    )
    close_location = float(feature["close_location"])
    close_quality = close_location if direction == Direction.LONG else 1.0 - close_location
    directional_bar = (
        bool(feature["bull_bar"]) if direction == Direction.LONG else bool(feature["bear_bar"])
    )
    directional_body = float(feature["body_ratio"]) if directional_bar else 0.0
    values = weights.model_dump()
    total = sum(values.values())
    quality = (
        values["context"] * context_quality
        + values["signal_close"] * close_quality
        + values["directional_body"] * directional_body
    ) / total
    return max(0.0, min(1.0, quality))


def feature_mapping(feature: Any) -> Mapping[str, Any]:
    """Convert a feature model, Series, or mapping into detector input."""
    if isinstance(feature, pd.Series):
        return feature.to_dict()
    if isinstance(feature, Mapping):
        return feature
    model_dump = getattr(feature, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        if isinstance(value, Mapping):
            return value
    raise TypeError("feature must be a mapping, pandas Series, or Pydantic model")


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"strategy configuration section {key!r} must be a mapping")
    return value
