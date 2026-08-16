"""Public event-driven historical backtest API."""

from brooks_trader.backtest.broker import (
    ClosedTrade,
    ExecutionEvent,
    ExecutionEventType,
    OpenPosition,
    PaperBroker,
    PendingEntry,
)
from brooks_trader.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    BacktestSettings,
    ExecutionSettings,
    RiskSettings,
)
from brooks_trader.backtest.metrics import BacktestMetrics, calculate_backtest_metrics
from brooks_trader.backtest.portfolio import Portfolio, PortfolioSnapshot
from brooks_trader.backtest.runner import run_backtest_experiment
from brooks_trader.backtest.trade_logger import trades_to_frame, write_trade_log

__all__ = [
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "BacktestSettings",
    "ClosedTrade",
    "ExecutionEvent",
    "ExecutionEventType",
    "ExecutionSettings",
    "OpenPosition",
    "PaperBroker",
    "PendingEntry",
    "Portfolio",
    "PortfolioSnapshot",
    "RiskSettings",
    "calculate_backtest_metrics",
    "run_backtest_experiment",
    "trades_to_frame",
    "write_trade_log",
]
