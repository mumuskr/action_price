"""Market regime classification and point-in-time context engine."""

from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd

from brooks_trader.market.always_in import AlwaysInTracker
from brooks_trader.market.trend import (
    ContextScores,
    MarketContextConfig,
    TrendScoreTracker,
    feature_mapping,
    load_market_context_config,
)
from brooks_trader.models.market_state import MarketRegime, MarketState


def classify_regime(trend_score: float, config: MarketContextConfig) -> MarketRegime:
    """Map a bounded score to a configured seven-state regime."""
    if not isfinite(trend_score) or not -1 <= trend_score <= 1:
        raise ValueError("trend_score must be finite and between -1 and 1")
    if trend_score >= config.strong_bull_threshold:
        return MarketRegime.STRONG_BULL_TREND
    if trend_score >= config.bull_threshold:
        return MarketRegime.BULL_TREND
    if trend_score > config.trading_range_threshold:
        return MarketRegime.WEAK_BULL
    if trend_score <= config.strong_bear_threshold:
        return MarketRegime.STRONG_BEAR_TREND
    if trend_score <= config.bear_threshold:
        return MarketRegime.BEAR_TREND
    if trend_score < -config.trading_range_threshold:
        return MarketRegime.WEAK_BEAR
    return MarketRegime.TRADING_RANGE


class MarketContextEngine:
    """Incrementally convert closed BarFeatures into traceable MarketState records."""

    def __init__(self, config: MarketContextConfig, *, strategy_version: str) -> None:
        if not strategy_version.strip():
            raise ValueError("strategy_version cannot be empty")
        self.config = config
        self.strategy_version = strategy_version
        self.reset()

    def reset(self) -> None:
        """Reset rolling scores and Always In state for a new independent stream."""
        self._score_tracker = TrendScoreTracker(self.config)
        self._always_in_tracker = AlwaysInTracker(self.config)
        self._last_bar_index: int | None = None
        self._last_timestamp: datetime | None = None

    def update(self, feature: Any) -> MarketState:
        """Consume exactly one closed feature row without future-bar access."""
        row = feature_mapping(feature)
        bar_index = int(row["bar_index"])
        timestamp = row["timestamp"]
        if not isinstance(timestamp, datetime):
            raise TypeError("feature timestamp must be a datetime")
        if self._last_bar_index is not None and bar_index <= self._last_bar_index:
            raise ValueError("feature bar_index must increase strictly")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("feature timestamp must increase strictly")

        scores = self._score_tracker.update(row)
        state = self._build_state(timestamp, bar_index, scores)
        self._last_bar_index = bar_index
        self._last_timestamp = timestamp
        return state

    def detect(self, features: pd.DataFrame) -> list[MarketState]:
        """Process a complete independent history through repeated ``update`` calls."""
        self.reset()
        return [self.update(row) for _, row in features.iterrows()]

    def _build_state(
        self,
        timestamp: datetime,
        bar_index: int,
        scores: ContextScores,
    ) -> MarketState:
        always_in = self._always_in_tracker.update(
            trend_score=scores.trend_score,
            ema_score=scores.ema_score,
            structure_score=scores.structure_score,
        )
        return MarketState(
            timestamp=timestamp,
            bar_index=bar_index,
            regime=classify_regime(scores.trend_score, self.config),
            always_in=always_in,
            trend_score=scores.trend_score,
            ema_score=scores.ema_score,
            structure_score=scores.structure_score,
            pressure_score=scores.pressure_score,
            overlap_score=scores.overlap_score,
            breakout_score=scores.breakout_score,
            strategy_version=self.strategy_version,
        )


__all__ = [
    "MarketContextConfig",
    "MarketContextEngine",
    "classify_regime",
    "load_market_context_config",
]
