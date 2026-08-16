"""Simple, confirmation-based Always In state machine."""

from dataclasses import dataclass

from brooks_trader.market.trend import MarketContextConfig
from brooks_trader.models.market_state import AlwaysInState


@dataclass
class AlwaysInTracker:
    """Track Always In direction without looking beyond the current bar."""

    config: MarketContextConfig
    state: AlwaysInState = AlwaysInState.NEUTRAL
    pending: AlwaysInState = AlwaysInState.NEUTRAL
    pending_bars: int = 0

    def update(
        self,
        *,
        trend_score: float,
        ema_score: float,
        structure_score: float,
    ) -> AlwaysInState:
        """Consume one score row and return the confirmed state at that row."""
        candidate = candidate_always_in(
            trend_score=trend_score,
            ema_score=ema_score,
            structure_score=structure_score,
            threshold=self.config.always_in_score_threshold,
        )
        if candidate == self.state:
            self.pending = AlwaysInState.NEUTRAL
            self.pending_bars = 0
            return self.state
        if candidate == AlwaysInState.NEUTRAL:
            self.pending = AlwaysInState.NEUTRAL
            self.pending_bars = 0
            return self.state
        if candidate == self.pending:
            self.pending_bars += 1
        else:
            self.pending = candidate
            self.pending_bars = 1
        if self.pending_bars >= self.config.always_in_confirmation_bars:
            self.state = candidate
            self.pending = AlwaysInState.NEUTRAL
            self.pending_bars = 0
        return self.state


def candidate_always_in(
    *,
    trend_score: float,
    ema_score: float,
    structure_score: float,
    threshold: float,
) -> AlwaysInState:
    """Return a directional candidate using transparent score alignment rules."""
    if trend_score >= threshold and ema_score > 0 and structure_score >= 0:
        return AlwaysInState.ALWAYS_IN_LONG
    if trend_score <= -threshold and ema_score < 0 and structure_score <= 0:
        return AlwaysInState.ALWAYS_IN_SHORT
    return AlwaysInState.NEUTRAL
