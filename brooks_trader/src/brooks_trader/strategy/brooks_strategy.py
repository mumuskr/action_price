"""Trading Engine decision layer for accepted Phase 5 setups."""

from brooks_trader.models import SetupEvaluation, SignalType, StrategySignal


class BrooksStrategy:
    """Convert accepted setups to signals without constructing broker orders."""

    def __init__(self, *, strategy_version: str) -> None:
        if not strategy_version.strip():
            raise ValueError("strategy_version cannot be empty")
        self.strategy_version = strategy_version

    def evaluate(self, evaluation: SetupEvaluation) -> StrategySignal | None:
        """Return a signal only for an accepted and version-matched setup."""
        if evaluation.strategy_version != self.strategy_version:
            raise ValueError("evaluation and strategy versions must match")
        if not evaluation.accepted:
            return None
        if evaluation.setup is None:
            raise RuntimeError("accepted evaluation is missing its setup")
        setup = evaluation.setup
        return StrategySignal(
            signal_type=SignalType.SECOND_ENTRY_WITH_TREND,
            created_at=evaluation.evaluated_at,
            signal_bar_index=setup.signal_bar_index,
            direction=setup.direction,
            setup=setup,
            reasons=list(setup.reasons),
            strategy_version=self.strategy_version,
        )
