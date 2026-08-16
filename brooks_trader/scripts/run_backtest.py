"""Run the event-driven backtester and Phase 7 empirical setup statistics."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from brooks_trader.backtest import BacktestEngine, write_trade_log  # noqa: E402
from brooks_trader.data import load_ohlcv  # noqa: E402
from brooks_trader.statistics import (  # noqa: E402
    calculate_setup_statistics,
    load_setup_statistics_config,
    write_setup_statistics,
)
from brooks_trader.strategy import StrategyModuleSelection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Configured market, for example SPY")
    parser.add_argument("--timeframe", required=True, help="Dataset timeframe, for example 5m")
    parser.add_argument(
        "--input",
        type=Path,
        help="Defaults to data/processed/symbol=<SYMBOL>/timeframe=<TIMEFRAME>/bars.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to the symbol/timeframe backtest partition's trades.parquet",
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        help="Defaults to the symbol/timeframe partition's setup_statistics.parquet",
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("config/strategy.yaml"),
    )
    parser.add_argument(
        "--markets-config",
        type=Path,
        default=Path("config/markets.yaml"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Replay only the first N bars; useful for a quick verification run",
    )
    parser.add_argument(
        "--disable-module",
        action="append",
        choices=tuple(StrategyModuleSelection.model_fields),
        default=[],
        help="Disable one executable strategy module; repeat to disable several",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.strip().upper()
    timeframe = args.timeframe.strip()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least one")
    source = args.input or (
        Path("data/processed") / f"symbol={symbol}" / f"timeframe={timeframe}" / "bars.parquet"
    )
    output_partition = Path("data/backtests") / f"symbol={symbol}" / f"timeframe={timeframe}"
    destination = args.output or (output_partition / "trades.parquet")
    statistics_destination = args.statistics_output or (
        output_partition / "setup_statistics.parquet"
    )
    frame = load_ohlcv(source)
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
    module_overrides = {module_id: False for module_id in args.disable_module}
    engine = BacktestEngine.from_config(
        symbol=symbol,
        timeframe=timeframe,
        strategy_path=args.strategy_config,
        markets_path=args.markets_config,
        module_overrides=module_overrides,
    )
    result = engine.run(frame)
    write_trade_log(list(result.trades), destination)
    statistics_config = load_setup_statistics_config(args.strategy_config)
    statistics = calculate_setup_statistics(result.trades, config=statistics_config)
    write_setup_statistics(statistics, statistics_destination)

    print(f"Backtest: {symbol} {timeframe}")
    print(f"Strategy version: {result.strategy_version}")
    print(f"Bars: {len(frame):,}")
    print(f"Patterns: {len(result.patterns):,}")
    print(f"Signals: {len(result.signals):,}")
    print(f"Trades: {len(result.trades):,}")
    print(f"Ending cash: {result.ending_cash:.2f}")
    print(f"Trade log: {destination}")
    print(f"Setup statistics: {statistics_destination}")
    print(f"Statistics rows: {len(statistics):,}")
    print(f"Metrics: {asdict(result.metrics)}")


if __name__ == "__main__":
    main()
