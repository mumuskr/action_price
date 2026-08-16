"""Validate a local OHLCV file and persist it as canonical Parquet."""

import argparse
from pathlib import Path
from typing import cast

from brooks_trader.data.loader import TimestampUnit, load_ohlcv
from brooks_trader.data.parquet_store import ParquetStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV or Parquet OHLCV file")
    parser.add_argument("--symbol", required=True, help="Market symbol, for example ES")
    parser.add_argument("--timeframe", required=True, help="Bar timeframe, for example 5m")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Root of the Hive-style Parquet dataset",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=("s", "ms", "us", "ns"),
        help="Required for integer epoch timestamps",
    )
    parser.add_argument(
        "--strategy-version",
        default="data-schema-v1",
        help="Version recorded in Parquet metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp_unit = cast(TimestampUnit | None, args.timestamp_unit)
    frame = load_ohlcv(args.input, timestamp_unit=timestamp_unit)
    destination = ParquetStore(args.output_root).write_bars(
        frame,
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_version=args.strategy_version,
    )
    print(f"Imported {len(frame)} bars to {destination}")


if __name__ == "__main__":
    main()
