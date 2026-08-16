"""Download validated US stock one-minute bars from a running IBKR TWS."""

import argparse
from datetime import datetime
from pathlib import Path

from brooks_trader.data import ParquetStore, download_us_stock_minute_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="US stock ticker, for example AAPL")
    parser.add_argument(
        "--primary-exchange",
        required=True,
        help="Primary exchange used to resolve the contract, for example NASDAQ or NYSE",
    )
    parser.add_argument("--duration", default="1 D", help="IBKR duration, for example '1 D'")
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        help="Optional timezone-aware ISO-8601 end time; current time when omitted",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=201)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = download_us_stock_minute_bars(
        args.symbol,
        primary_exchange=args.primary_exchange,
        duration=args.duration,
        end=args.end,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )
    symbol = metadata["symbol"]
    destination = ParquetStore(args.output_root).write_bars(
        frame,
        symbol=symbol,
        timeframe="1m",
        strategy_version="ibkr-historical-v1",
        source_metadata=metadata,
    )
    print(
        f"Downloaded {len(frame)} {symbol} bars from {frame['timestamp'].iloc[0].isoformat()} "
        f"through {frame['timestamp'].iloc[-1].isoformat()} to {destination}"
    )


if __name__ == "__main__":
    main()
