"""Explicit bearish-pullback L1/L2 state machine."""

from enum import StrEnum
from typing import Any

from brooks_trader.models import Bar, Direction, MarketState, PatternEvent, PatternType
from brooks_trader.patterns.base import (
    PatternDetectorConfig,
    StatefulPatternDetector,
    is_short_pattern_context,
)


class BearPullbackState(StrEnum):
    IDLE = "IDLE"
    PULLBACK_STARTED = "PULLBACK_STARTED"
    L1_TRIGGERED = "L1_TRIGGERED"
    L1_FAILED_OR_NEW_PULLBACK = "L1_FAILED_OR_NEW_PULLBACK"
    L2_READY = "L2_READY"


class L1L2Detector(StatefulPatternDetector):
    """Detect first and second downward attempts inside a bear pullback."""

    detector_name = "L1L2Detector"

    def __init__(self, config: PatternDetectorConfig, *, strategy_version: str) -> None:
        super().__init__(config, strategy_version=strategy_version)
        self.reset()

    def reset(self) -> None:
        self.reset_base()
        self.state = BearPullbackState.IDLE
        self.start_index: int | None = None
        self.pullback_bars = 0

    def update(
        self,
        bar: Bar,
        feature: Any,
        context: MarketState,
    ) -> list[PatternEvent]:
        row, previous = self.begin_update(bar, feature, context)
        bar_index = int(row["bar_index"])
        events: list[PatternEvent] = []

        if not is_short_pattern_context(context):
            self._reset_state(bar, bar_index, "short_context_lost")
            return events
        if self.state != BearPullbackState.IDLE:
            self.pullback_bars += 1
            if self.pullback_bars > self.config.pullback_max_bars:
                self._reset_state(bar, bar_index, "pullback_expired")
                return events

        if self.state == BearPullbackState.IDLE:
            if bool(row["bull_bar"]) or bool(row["higher_high"]):
                self.start_index = bar_index
                self.pullback_bars = 1
                self._set_state(
                    bar,
                    bar_index,
                    BearPullbackState.PULLBACK_STARTED,
                    "bull_bar_or_higher_high",
                )
            return events

        if self.state == BearPullbackState.PULLBACK_STARTED:
            if (
                previous is not None
                and self.pullback_bars >= self.config.pullback_min_bars
                and bar.low < previous.low
            ):
                before = self.state
                event = self._event(PatternType.L1, bar, row, context, 1, before)
                events.append(event)
                self._set_state(
                    bar,
                    bar_index,
                    BearPullbackState.L1_TRIGGERED,
                    "first_down_attempt",
                    PatternType.L1,
                )
            return events

        if self.state == BearPullbackState.L1_TRIGGERED:
            if previous is not None and bar.high > previous.high:
                self._set_state(
                    bar,
                    bar_index,
                    BearPullbackState.L1_FAILED_OR_NEW_PULLBACK,
                    "new_up_leg_after_l1",
                )
            return events

        if self.state == BearPullbackState.L1_FAILED_OR_NEW_PULLBACK:
            self._set_state(
                bar,
                bar_index,
                BearPullbackState.L2_READY,
                "second_attempt_armed",
            )

        if (
            self.state == BearPullbackState.L2_READY
            and previous is not None
            and bar.low < previous.low
        ):
            before = self.state
            event = self._event(PatternType.L2, bar, row, context, 2, before)
            events.append(event)
            self._reset_state(bar, bar_index, "second_down_attempt", PatternType.L2)
        return events

    def _event(
        self,
        pattern_type: PatternType,
        bar: Bar,
        feature: Any,
        context: MarketState,
        attempt_number: int,
        state_before: BearPullbackState,
    ) -> PatternEvent:
        if self.start_index is None:
            raise RuntimeError("pullback start_index is unavailable")
        return self.build_event(
            pattern_type=pattern_type,
            direction=Direction.SHORT,
            bar=bar,
            feature=feature,
            context=context,
            start_index=self.start_index,
            pullback_bars=self.pullback_bars,
            attempt_number=attempt_number,
            state_before=state_before,
        )

    def _set_state(
        self,
        bar: Bar,
        bar_index: int,
        state: BearPullbackState,
        condition: str,
        pattern_type: PatternType | None = None,
    ) -> None:
        before = self.state
        self.state = state
        self.record_transition(
            bar=bar,
            bar_index=bar_index,
            state_before=before,
            condition=condition,
            state_after=state,
            pattern_type=pattern_type,
        )

    def _reset_state(
        self,
        bar: Bar,
        bar_index: int,
        condition: str,
        pattern_type: PatternType | None = None,
    ) -> None:
        if self.state != BearPullbackState.IDLE:
            self._set_state(
                bar,
                bar_index,
                BearPullbackState.IDLE,
                condition,
                pattern_type,
            )
        self.start_index = None
        self.pullback_bars = 0
