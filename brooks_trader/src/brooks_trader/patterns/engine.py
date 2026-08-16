"""Orchestration for the four Phase 4 first/second-entry detectors."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from brooks_trader.models import Bar, MarketState, PatternEvent
from brooks_trader.patterns.base import DetectorTransition, PatternDetectorConfig
from brooks_trader.patterns.h1_h2 import H1H2Detector
from brooks_trader.patterns.l1_l2 import L1L2Detector


class FirstSecondEntryPatternEngine:
    """Run bullish and bearish detectors without creating trade decisions."""

    def __init__(self, config: PatternDetectorConfig, *, strategy_version: str) -> None:
        self.long_detector = H1H2Detector(config, strategy_version=strategy_version)
        self.short_detector = L1L2Detector(config, strategy_version=strategy_version)

    @property
    def debug_log(self) -> list[DetectorTransition]:
        """Return both detector logs in chronological order."""
        return sorted(
            self.long_detector.debug_log + self.short_detector.debug_log,
            key=lambda transition: (transition.bar_index, transition.detector),
        )

    def reset(self) -> None:
        self.long_detector.reset()
        self.short_detector.reset()

    def update(
        self,
        bar: Bar,
        feature: Any,
        context: MarketState,
    ) -> list[PatternEvent]:
        """Consume one synchronized observation and return zero or more patterns."""
        return self.long_detector.update(bar, feature, context) + self.short_detector.update(
            bar,
            feature,
            context,
        )

    def detect(
        self,
        bars: Sequence[Bar],
        features: pd.DataFrame,
        contexts: Sequence[MarketState],
    ) -> list[PatternEvent]:
        """Run an independent history through the same incremental update path."""
        if len(bars) != len(features) or len(bars) != len(contexts):
            raise ValueError("bars, features, and contexts must have equal lengths")
        self.reset()
        events: list[PatternEvent] = []
        for position, bar in enumerate(bars):
            events.extend(self.update(bar, features.iloc[position], contexts[position]))
        return events
