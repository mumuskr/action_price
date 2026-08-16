from __future__ import annotations

from pathlib import Path

import pandas as pd

from brooks_trader.backtest import BacktestEngine, trades_to_frame, write_trade_log

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_trade_log_round_trip_preserves_utc_timestamps(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "data/processed/symbol=SPY/timeframe=5m/bars.parquet"
    engine = BacktestEngine.from_config(
        symbol="SPY",
        timeframe="5m",
        strategy_path=PROJECT_ROOT / "config/strategy.yaml",
        markets_path=PROJECT_ROOT / "config/markets.yaml",
    )
    result = engine.run(pd.read_parquet(source).iloc[:3000])
    destination = write_trade_log(list(result.trades), tmp_path / "trades.parquet")
    restored = pd.read_parquet(destination)

    assert not restored.empty
    assert str(restored["entry_time"].dtype) == "datetime64[us, UTC]"
    assert str(restored["exit_time"].dtype) == "datetime64[us, UTC]"
    assert set(restored["strategy_version"]) == {"0.7.0-phase7"}
    assert restored["market_state"].str.contains('"trend_score"').all()


def test_empty_trade_log_keeps_stable_schema() -> None:
    frame = trades_to_frame([])

    assert frame.empty
    assert "strategy_version" in frame.columns
    assert "mfe_r" in frame.columns
