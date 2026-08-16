"""Event-driven, bar-by-bar backtest orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from brooks_trader.backtest.broker import (
    ExecutionEvent,
    ExecutionEventType,
    PaperBroker,
)
from brooks_trader.backtest.metrics import BacktestMetrics, calculate_backtest_metrics
from brooks_trader.backtest.portfolio import Portfolio
from brooks_trader.data.loader import bars_from_frame, normalize_ohlcv
from brooks_trader.features import (
    BarFeatureConfig,
    calculate_bar_features,
    load_bar_feature_config,
)
from brooks_trader.market import (
    MarketContextConfig,
    MarketContextEngine,
    load_market_context_config,
)
from brooks_trader.models import (
    MarketState,
    Order,
    PatternEvent,
    SetupEvaluation,
    StrategySignal,
    TradeRecord,
)
from brooks_trader.patterns import (
    FirstSecondEntryPatternEngine,
    PatternDetectorConfig,
    load_pattern_detector_config,
)
from brooks_trader.strategy import (
    BrooksStrategy,
    SetupEngine,
    SetupEngineConfig,
    load_setup_engine_config,
)
from brooks_trader.strategy.catalog import StrategyModuleSelection


class BacktestSettings(BaseModel):
    """Scope controls for the Phase 6 single-position backtester."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_cash: float = Field(gt=0)
    allow_multiple_positions: Literal[False]
    pending_order_expiry_bars: int = Field(ge=1)
    close_open_position_at_end: bool


class RiskSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    risk_per_trade: float = Field(gt=0, le=1)


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slippage_ticks: int = Field(ge=0)
    commission_per_trade: float = Field(ge=0)
    same_bar_stop_target_policy: Literal["adverse", "favorable"]


@dataclass(frozen=True)
class BacktestResult:
    """All traceable artifacts produced by one independent historical replay."""

    symbol: str
    timeframe: str
    strategy_version: str
    features: pd.DataFrame
    market_states: tuple[MarketState, ...]
    patterns: tuple[PatternEvent, ...]
    setup_evaluations: tuple[SetupEvaluation, ...]
    signals: tuple[StrategySignal, ...]
    orders: tuple[Order, ...]
    order_history: tuple[Order, ...]
    executions: tuple[ExecutionEvent, ...]
    trades: tuple[TradeRecord, ...]
    metrics: BacktestMetrics
    ending_cash: float


class BacktestEngine:
    """Replay one symbol/timeframe using only information known at each bar.

    Processing order is intentional: outstanding orders and open positions see the
    current bar before that bar's close can create a new signal. Therefore a signal on
    bar N cannot fill until bar N+1 or later.
    """

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy_version: str,
        feature_config: BarFeatureConfig,
        context_config: MarketContextConfig,
        pattern_config: PatternDetectorConfig,
        setup_config: SetupEngineConfig,
        risk_settings: RiskSettings,
        execution_settings: ExecutionSettings,
        backtest_settings: BacktestSettings,
    ) -> None:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol cannot be empty")
        if not timeframe.strip():
            raise ValueError("timeframe cannot be empty")
        if setup_config.market.symbol != normalized_symbol:
            raise ValueError("setup market and backtest symbol must match")
        if not strategy_version.strip():
            raise ValueError("strategy_version cannot be empty")
        self.symbol = normalized_symbol
        self.timeframe = timeframe.strip()
        self.strategy_version = strategy_version
        self.feature_config = feature_config
        self.context_config = context_config
        self.pattern_config = pattern_config
        self.setup_config = setup_config
        self.risk_settings = risk_settings
        self.execution_settings = execution_settings
        self.backtest_settings = backtest_settings

    @classmethod
    def from_config(
        cls,
        *,
        symbol: str,
        timeframe: str,
        strategy_path: str | Path,
        markets_path: str | Path,
        module_overrides: Mapping[str, object] | StrategyModuleSelection | None = None,
    ) -> BacktestEngine:
        """Build all Phase 2-6 components from the versioned YAML configuration."""
        feature_config = load_bar_feature_config(strategy_path)
        context_config, context_version = load_market_context_config(strategy_path)
        pattern_config, pattern_version = load_pattern_detector_config(strategy_path)
        setup_config, setup_version = load_setup_engine_config(
            strategy_path,
            markets_path,
            symbol=symbol,
            module_overrides=module_overrides,
        )
        if len({context_version, pattern_version, setup_version}) != 1:
            raise ValueError("all engines must use the same strategy version")
        with Path(strategy_path).expanduser().open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("strategy configuration must be a mapping")
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            strategy_version=setup_version,
            feature_config=feature_config,
            context_config=context_config,
            pattern_config=pattern_config,
            setup_config=setup_config,
            risk_settings=RiskSettings.model_validate(raw.get("risk")),
            execution_settings=ExecutionSettings.model_validate(raw.get("execution")),
            backtest_settings=BacktestSettings.model_validate(raw.get("backtest")),
        )

    def run(
        self,
        frame: pd.DataFrame,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        """Run an independent historical replay and return every audit artifact."""
        normalized = normalize_ohlcv(frame)
        bars = bars_from_frame(normalized)
        total_bars = len(bars)
        if progress_callback is not None:
            progress_callback(0, total_bars)
        features = calculate_bar_features(normalized, config=self.feature_config)
        context_engine = MarketContextEngine(
            self.context_config,
            strategy_version=self.strategy_version,
        )
        pattern_engine = FirstSecondEntryPatternEngine(
            self.pattern_config,
            strategy_version=self.strategy_version,
        )
        setup_engine = SetupEngine(
            self.setup_config,
            strategy_version=self.strategy_version,
        )
        strategy = BrooksStrategy(strategy_version=self.strategy_version)
        broker = PaperBroker(
            tick_size=self.setup_config.market.tick_size,
            slippage_ticks=self.execution_settings.slippage_ticks,
            commission_per_trade=self.execution_settings.commission_per_trade,
            same_bar_stop_target_policy=(self.execution_settings.same_bar_stop_target_policy),
            pending_order_expiry_bars=self.backtest_settings.pending_order_expiry_bars,
        )
        portfolio = Portfolio(
            initial_cash=self.backtest_settings.initial_cash,
            risk_per_trade=self.risk_settings.risk_per_trade,
            point_value=self.setup_config.market.point_value,
            commission_per_trade=self.execution_settings.commission_per_trade,
            slippage_ticks=self.execution_settings.slippage_ticks,
        )

        contexts: list[MarketState] = []
        patterns: list[PatternEvent] = []
        evaluations: list[SetupEvaluation] = []
        signals: list[StrategySignal] = []
        executions: list[ExecutionEvent] = []
        trades: list[TradeRecord] = []

        for bar_index, bar in enumerate(bars):
            self._consume_executions(
                broker.process_bar(bar, bar_index=bar_index),
                portfolio,
                executions,
                trades,
            )
            feature = features.iloc[bar_index]
            context = context_engine.update(feature)
            contexts.append(context)
            current_patterns = pattern_engine.update(bar, feature, context)
            current_patterns = [
                pattern
                for pattern in current_patterns
                if (pattern.pattern_type.value == "H2" and self.setup_config.modules.h2_with_trend)
                or (pattern.pattern_type.value == "L2" and self.setup_config.modules.l2_with_trend)
                or pattern.pattern_type.value not in {"H2", "L2"}
            ]
            patterns.extend(current_patterns)
            for pattern in current_patterns:
                evaluation = setup_engine.evaluate(
                    pattern,
                    bars[: bar_index + 1],
                    features.iloc[: bar_index + 1],
                    contexts,
                )
                evaluations.append(evaluation)
                signal = strategy.evaluate(evaluation)
                if signal is None:
                    continue
                signals.append(signal)
                if broker.position is not None or broker.pending_entry is not None:
                    continue
                quantity = portfolio.size_for_entry(
                    entry_price=signal.setup.entry,
                    stop_price=signal.setup.initial_stop,
                )
                if quantity > 0:
                    broker.submit_signal(
                        signal,
                        quantity=quantity,
                        submitted_index=bar_index,
                    )
            if progress_callback is not None:
                progress_callback(bar_index + 1, total_bars)

        if self.backtest_settings.close_open_position_at_end:
            self._consume_executions(
                broker.force_close(bars[-1], bar_index=len(bars) - 1),
                portfolio,
                executions,
                trades,
            )
        metrics = calculate_backtest_metrics(
            trades,
            initial_cash=self.backtest_settings.initial_cash,
        )
        return BacktestResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_version=self.strategy_version,
            features=features,
            market_states=tuple(contexts),
            patterns=tuple(patterns),
            setup_evaluations=tuple(evaluations),
            signals=tuple(signals),
            orders=broker.orders,
            order_history=broker.order_history,
            executions=tuple(executions),
            trades=tuple(trades),
            metrics=metrics,
            ending_cash=portfolio.cash,
        )

    def _consume_executions(
        self,
        current: list[ExecutionEvent],
        portfolio: Portfolio,
        executions: list[ExecutionEvent],
        trades: list[TradeRecord],
    ) -> None:
        executions.extend(current)
        for event in current:
            if event.event_type != ExecutionEventType.TRADE_CLOSED:
                continue
            if event.trade is None:
                raise RuntimeError("closed execution is missing its trade")
            pnl = portfolio.realize(event.trade)
            trades.append(
                portfolio.build_trade_record(
                    event.trade,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    pnl=pnl,
                )
            )
