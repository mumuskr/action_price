"""Basic portfolio-level metrics for completed Phase 6 backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from brooks_trader.models import TradeRecord


@dataclass(frozen=True)
class BacktestMetrics:
    """Small, explicit metric set; conditional setup statistics belong to Phase 7."""

    total_trades: int
    wins: int
    losses: int
    win_rate: float | None
    net_pnl: float
    total_return: float
    expectancy_r: float | None
    profit_factor: float | None
    max_drawdown: float


def calculate_backtest_metrics(
    trades: list[TradeRecord],
    *,
    initial_cash: float,
) -> BacktestMetrics:
    """Calculate metrics from closed trades in chronological realization order."""
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    pnls = [float(trade.pnl or 0.0) for trade in trades]
    pnl_rs = [float(trade.pnl_r or 0.0) for trade in trades]
    wins = sum(pnl > 0 for pnl in pnls)
    losses = sum(pnl < 0 for pnl in pnls)
    count = len(trades)
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
    if gross_loss > 0:
        profit_factor: float | None = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = inf
    else:
        profit_factor = None
    net_pnl = sum(pnls)
    return BacktestMetrics(
        total_trades=count,
        wins=wins,
        losses=losses,
        win_rate=wins / count if count else None,
        net_pnl=net_pnl,
        total_return=net_pnl / initial_cash,
        expectancy_r=sum(pnl_rs) / count if count else None,
        profit_factor=profit_factor,
        max_drawdown=_max_drawdown(pnls, initial_cash),
    )


def _max_drawdown(pnls: list[float], initial_cash: float) -> float:
    equity = initial_cash
    peak = initial_cash
    maximum = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum
