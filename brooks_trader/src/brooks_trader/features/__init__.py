"""Public Phase 2 bar-feature API."""

from brooks_trader.features.bars import (
    BarFeatureConfig,
    calculate_bar_features,
    calculate_body_ratio,
    calculate_tail_ratio,
    is_bear_bar,
    is_bull_bar,
    is_doji,
    is_inside_bar,
    is_outside_bar,
    is_trend_bar,
    load_bar_feature_config,
)
from brooks_trader.features.ema import calculate_ema20, calculate_ema_slope
from brooks_trader.features.overlap import calculate_overlap

__all__ = [
    "BarFeatureConfig",
    "calculate_bar_features",
    "calculate_body_ratio",
    "calculate_ema20",
    "calculate_ema_slope",
    "calculate_overlap",
    "calculate_tail_ratio",
    "is_bear_bar",
    "is_bull_bar",
    "is_doji",
    "is_inside_bar",
    "is_outside_bar",
    "is_trend_bar",
    "load_bar_feature_config",
]
