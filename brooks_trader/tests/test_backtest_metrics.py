from types import SimpleNamespace
from typing import cast

import pytest

from brooks_trader.backtest import calculate_backtest_metrics
from brooks_trader.models import TradeRecord


def test_basic_metrics_include_expectancy_profit_factor_and_drawdown() -> None:
    trades = cast(
        list[TradeRecord],
        [
            SimpleNamespace(pnl=100.0, pnl_r=1.0),
            SimpleNamespace(pnl=-50.0, pnl_r=-0.5),
            SimpleNamespace(pnl=-25.0, pnl_r=-0.25),
        ],
    )

    metrics = calculate_backtest_metrics(trades, initial_cash=1000)

    assert metrics.total_trades == 3
    assert metrics.win_rate == pytest.approx(1 / 3)
    assert metrics.net_pnl == 25
    assert metrics.total_return == 0.025
    assert metrics.expectancy_r == pytest.approx(1 / 12)
    assert metrics.profit_factor == pytest.approx(4 / 3)
    assert metrics.max_drawdown == 75
