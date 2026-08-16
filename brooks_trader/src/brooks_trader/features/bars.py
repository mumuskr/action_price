"""Causal bar-feature calculations using current and previously closed bars only."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from brooks_trader.data.loader import normalize_ohlcv
from brooks_trader.features.ema import calculate_ema20, calculate_ema_slope


class BarFeatureConfig(BaseModel):
    """Validated parameters used by the Phase 2 feature engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doji_body_ratio: float = Field(ge=0, le=1)
    strong_body_ratio: float = Field(ge=0, le=1)
    strong_close_threshold: float = Field(ge=0.5, le=1)
    minimum_range: float = Field(ge=0)
    ema_period: int = Field(ge=1)
    ema_slope_lookback: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_body_threshold_order(self) -> "BarFeatureConfig":
        if self.doji_body_ratio >= self.strong_body_ratio:
            raise ValueError("doji_body_ratio must be below strong_body_ratio")
        return self


def load_bar_feature_config(path: str | Path) -> BarFeatureConfig:
    """Load only the Phase 2 parameters from a strategy YAML file."""
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"strategy configuration must be a mapping: {source}")
    bar = _require_mapping(raw, "bar")
    trend = _require_mapping(raw, "trend")
    return BarFeatureConfig.model_validate(
        {
            "doji_body_ratio": bar.get("doji_body_ratio"),
            "strong_body_ratio": bar.get("strong_body_ratio"),
            "strong_close_threshold": bar.get("strong_close_threshold"),
            "minimum_range": bar.get("minimum_range"),
            "ema_period": trend.get("ema_period"),
            "ema_slope_lookback": trend.get("ema_slope_lookback"),
        }
    )


def calculate_bar_features(frame: pd.DataFrame, *, config: BarFeatureConfig) -> pd.DataFrame:
    """Calculate a feature row for every OHLCV row without future-bar access.

    BROOKS_CONCEPT: trend bars, dojis, inside/outside bars, and overlapping bars.
    COMPUTATIONAL_PROXY: the exact configurable definitions implemented below. They are
    research rules, not formulas claimed to appear in Brooks' books.
    """
    bars = normalize_ohlcv(frame)
    bar_range = bars["high"] - bars["low"]
    body = (bars["close"] - bars["open"]).abs()
    body_ratio = _safe_ratio(body, bar_range, zero_denominator=0.0)
    upper_tail = bars["high"] - bars[["open", "close"]].max(axis=1)
    lower_tail = bars[["open", "close"]].min(axis=1) - bars["low"]
    upper_tail_ratio = _safe_ratio(upper_tail, bar_range, zero_denominator=0.0)
    lower_tail_ratio = _safe_ratio(lower_tail, bar_range, zero_denominator=0.0)
    close_location = _safe_ratio(
        bars["close"] - bars["low"],
        bar_range,
        zero_denominator=0.5,
    )

    bull_bar = bars["close"] > bars["open"]
    bear_bar = bars["close"] < bars["open"]
    valid_range = (bar_range > 0) & (bar_range >= config.minimum_range)
    doji = (~valid_range) | (body_ratio <= config.doji_body_ratio)
    trend_bar = (
        valid_range
        & (body_ratio >= config.strong_body_ratio)
        & (
            (bull_bar & (close_location >= config.strong_close_threshold))
            | (bear_bar & (close_location <= 1 - config.strong_close_threshold))
        )
    )

    previous_high = bars["high"].shift(1)
    previous_low = bars["low"].shift(1)
    has_previous = previous_high.notna()
    equal_range = (bars["high"] == previous_high) & (bars["low"] == previous_low)
    inside_bar = (
        has_previous
        & (bars["high"] <= previous_high)
        & (bars["low"] >= previous_low)
        & ~equal_range
    )
    outside_bar = (
        has_previous
        & (bars["high"] >= previous_high)
        & (bars["low"] <= previous_low)
        & ~equal_range
    )

    overlap_high = pd.concat([bars["high"], previous_high], axis=1).min(axis=1)
    overlap_low = pd.concat([bars["low"], previous_low], axis=1).max(axis=1)
    intersection = (overlap_high - overlap_low).clip(lower=0)
    union = pd.concat([bars["high"], previous_high], axis=1).max(axis=1) - pd.concat(
        [bars["low"], previous_low], axis=1
    ).min(axis=1)
    overlap_previous = _safe_ratio(intersection, union, zero_denominator=1.0).where(
        has_previous,
        0.0,
    )

    ema20 = calculate_ema20(bars["close"], period=config.ema_period)
    ema_slope = calculate_ema_slope(ema20, lookback=config.ema_slope_lookback).astype("Float64")

    return pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "bar_index": np.arange(len(bars), dtype=np.int64),
            "range": bar_range,
            "body": body,
            "body_ratio": body_ratio,
            "upper_tail": upper_tail,
            "lower_tail": lower_tail,
            "upper_tail_ratio": upper_tail_ratio,
            "lower_tail_ratio": lower_tail_ratio,
            "close_location": close_location,
            "bull_bar": bull_bar,
            "bear_bar": bear_bar,
            "trend_bar": trend_bar,
            "doji": doji,
            "inside_bar": inside_bar,
            "outside_bar": outside_bar,
            "higher_high": has_previous & (bars["high"] > previous_high),
            "higher_low": has_previous & (bars["low"] > previous_low),
            "lower_high": has_previous & (bars["high"] < previous_high),
            "lower_low": has_previous & (bars["low"] < previous_low),
            "overlap_previous": overlap_previous,
            "ema20": ema20,
            "distance_to_ema": bars["close"] - ema20,
            "ema_slope": ema_slope,
        }
    )


def calculate_body_ratio(open_price: float, close_price: float, bar_range: float) -> float:
    """Return absolute body size as a share of the full bar range."""
    _require_nonnegative_range(bar_range)
    return 0.0 if bar_range == 0 else abs(close_price - open_price) / bar_range


def calculate_tail_ratio(tail: float, bar_range: float) -> float:
    """Return one tail's size as a share of the full bar range."""
    _require_nonnegative_range(bar_range)
    if tail < 0:
        raise ValueError("tail cannot be negative")
    return 0.0 if bar_range == 0 else tail / bar_range


def is_bull_bar(open_price: float, close_price: float) -> bool:
    """Return whether the bar closed strictly above its open."""
    return close_price > open_price


def is_bear_bar(open_price: float, close_price: float) -> bool:
    """Return whether the bar closed strictly below its open."""
    return close_price < open_price


def is_doji(body_ratio: float, *, threshold: float) -> bool:
    """Classify a doji using the configured body/range threshold."""
    _require_unit_interval(body_ratio, "body_ratio")
    _require_unit_interval(threshold, "threshold")
    return body_ratio <= threshold


def is_trend_bar(
    *,
    body_ratio: float,
    close_location: float,
    bull_bar: bool,
    bear_bar: bool,
    strong_body_ratio: float,
    strong_close_threshold: float,
) -> bool:
    """Classify a directional strong-body, strong-close bar."""
    for name, value in (
        ("body_ratio", body_ratio),
        ("close_location", close_location),
        ("strong_body_ratio", strong_body_ratio),
        ("strong_close_threshold", strong_close_threshold),
    ):
        _require_unit_interval(value, name)
    return body_ratio >= strong_body_ratio and (
        (bull_bar and close_location >= strong_close_threshold)
        or (bear_bar and close_location <= 1 - strong_close_threshold)
    )


def is_inside_bar(
    current_high: float,
    current_low: float,
    previous_high: float,
    previous_low: float,
) -> bool:
    """Return whether the current range is strictly nested in the previous range."""
    _validate_adjacent_ranges(current_high, current_low, previous_high, previous_low)
    return (
        current_high <= previous_high
        and current_low >= previous_low
        and (current_high < previous_high or current_low > previous_low)
    )


def is_outside_bar(
    current_high: float,
    current_low: float,
    previous_high: float,
    previous_low: float,
) -> bool:
    """Return whether the current range strictly contains the previous range."""
    _validate_adjacent_ranges(current_high, current_low, previous_high, previous_low)
    return (
        current_high >= previous_high
        and current_low <= previous_low
        and (current_high > previous_high or current_low < previous_low)
    )


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    zero_denominator: float,
) -> pd.Series:
    result = numerator.div(denominator.where(denominator != 0))
    return result.fillna(zero_denominator).clip(lower=0.0, upper=1.0)


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"strategy configuration section {key!r} must be a mapping")
    return value


def _require_nonnegative_range(bar_range: float) -> None:
    if bar_range < 0:
        raise ValueError("bar_range cannot be negative")


def _require_unit_interval(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_adjacent_ranges(
    current_high: float,
    current_low: float,
    previous_high: float,
    previous_low: float,
) -> None:
    if current_high < current_low or previous_high < previous_low:
        raise ValueError("bar high cannot be below bar low")
