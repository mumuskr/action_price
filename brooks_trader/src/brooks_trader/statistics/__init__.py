"""Empirical setup statistics and excursion calculations."""

from brooks_trader.statistics.conditional_probability import find_empirical_probability
from brooks_trader.statistics.mfe_mae import Excursion, calculate_mfe_mae
from brooks_trader.statistics.setup_stats import (
    EmaSlopeBucket,
    SessionBucket,
    SetupStatistics,
    SetupStatisticsConfig,
    SignalQualityBucket,
    StatisticsScope,
    TradeConditions,
    VolatilityRegime,
    calculate_setup_statistics,
    classify_trade_conditions,
    load_setup_statistics_config,
    statistics_to_frame,
    write_setup_statistics,
)

__all__ = [
    "EmaSlopeBucket",
    "Excursion",
    "SessionBucket",
    "SetupStatistics",
    "SetupStatisticsConfig",
    "SignalQualityBucket",
    "StatisticsScope",
    "TradeConditions",
    "VolatilityRegime",
    "calculate_mfe_mae",
    "calculate_setup_statistics",
    "classify_trade_conditions",
    "find_empirical_probability",
    "load_setup_statistics_config",
    "statistics_to_frame",
    "write_setup_statistics",
]
