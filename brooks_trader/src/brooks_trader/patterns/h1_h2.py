"""Explicit bullish-pullback H1/H2 state machine."""

from enum import StrEnum
from typing import Any

from brooks_trader.models import Bar, Direction, MarketState, PatternEvent, PatternType
from brooks_trader.patterns.base import (
    PatternDetectorConfig,
    StatefulPatternDetector,
    is_long_pattern_context,
)


class BullPullbackState(StrEnum):
    IDLE = "IDLE"
    PULLBACK_STARTED = "PULLBACK_STARTED"
    H1_TRIGGERED = "H1_TRIGGERED"
    H1_FAILED_OR_NEW_PULLBACK = "H1_FAILED_OR_NEW_PULLBACK"
    H2_READY = "H2_READY"


class H1H2Detector(StatefulPatternDetector):
    """Detect first and second upward attempts inside a bull pullback."""

    detector_name = "H1H2Detector"

    def __init__(self, config: PatternDetectorConfig, *, strategy_version: str) -> None:
        super().__init__(config, strategy_version=strategy_version)
        self.reset()

    def reset(self) -> None:
        self.reset_base()
        self.state = BullPullbackState.IDLE
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

        if not is_long_pattern_context(context):
            self._reset_state(bar, bar_index, "long_context_lost")
            return events
        if self.state != BullPullbackState.IDLE:
            self.pullback_bars += 1
            if self.pullback_bars > self.config.pullback_max_bars:
                self._reset_state(bar, bar_index, "pullback_expired")
                return events

        if self.state == BullPullbackState.IDLE:
            if bool(row["bear_bar"]) or bool(row["lower_low"]):
                self.start_index = bar_index
                self.pullback_bars = 1
                self._set_state(
                    bar,
                    bar_index,
                    BullPullbackState.PULLBACK_STARTED,
                    "bear_bar_or_lower_low",
                )
            return events

        if self.state == BullPullbackState.PULLBACK_STARTED:
            if (
                previous is not None
                and self.pullback_bars >= self.config.pullback_min_bars
                and bar.high > previous.high
            ):
                before = self.state
                event = self._event(PatternType.H1, bar, row, context, 1, before)
                events.append(event)
                self._set_state(
                    bar,
                    bar_index,
                    BullPullbackState.H1_TRIGGERED,
                    "first_up_attempt",
                    PatternType.H1,
                )
            return events

        if self.state == BullPullbackState.H1_TRIGGERED:
            if previous is not None and bar.low < previous.low:
                self._set_state(
                    bar,
                    bar_index,
                    BullPullbackState.H1_FAILED_OR_NEW_PULLBACK,
                    "new_down_leg_after_h1",
                )
            return events

        if self.state == BullPullbackState.H1_FAILED_OR_NEW_PULLBACK:
            self._set_state(
                bar,
                bar_index,
                BullPullbackState.H2_READY,
                "second_attempt_armed",
            )

        if (
            self.state == BullPullbackState.H2_READY
            and previous is not None
            and bar.high > previous.high
        ):
            before = self.state
            event = self._event(PatternType.H2, bar, row, context, 2, before)
            events.append(event)
            self._reset_state(bar, bar_index, "second_up_attempt", PatternType.H2)
        return events

    def _event(
        self,
        pattern_type: PatternType,
        bar: Bar,
        feature: Any,
        context: MarketState,
        attempt_number: int,
        state_before: BullPullbackState,
    ) -> PatternEvent:
        if self.start_index is None:
            raise RuntimeError("pullback start_index is unavailable")
        return self.build_event(
            pattern_type=pattern_type,
            direction=Direction.LONG,
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
        state: BullPullbackState,
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
        if self.state != BullPullbackState.IDLE:
            self._set_state(
                bar,
                bar_index,
                BullPullbackState.IDLE,
                condition,
                pattern_type,
            )
        self.start_index = None
        self.pullback_bars = 0
