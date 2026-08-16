"""Lookup of sample-qualified empirical probabilities for Trader's Equation."""

from __future__ import annotations

from collections.abc import Sequence

from brooks_trader.models import MarketRegime, PatternType
from brooks_trader.statistics.setup_stats import SetupStatistics, StatisticsScope


def find_empirical_probability(
    statistics: Sequence[SetupStatistics],
    *,
    symbol: str,
    timeframe: str,
    strategy_version: str,
    pattern_type: PatternType,
    market_regime: MarketRegime,
) -> float | None:
    """Return an exact pattern/regime probability or ``None`` when unsupported."""
    matches = [
        statistic
        for statistic in statistics
        if statistic.scope == StatisticsScope.PATTERN_REGIME
        and statistic.symbol == symbol
        and statistic.timeframe == timeframe
        and statistic.strategy_version == strategy_version
        and statistic.pattern_type == pattern_type
        and statistic.market_regime == market_regime
    ]
    if len(matches) > 1:
        raise ValueError("duplicate conditional statistics for the requested context")
    return matches[0].probability_win if matches else None
