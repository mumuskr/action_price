"""Download ten years of US index ETF minute bars from IBKR and aggregate them."""

import argparse
import copy
import json
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brooks_trader.data.ibkr_history import IBKRHistoricalClient
from brooks_trader.data.loader import normalize_ohlcv, read_parquet_frame
from brooks_trader.data.us_market_bars import (
    regular_session_minute_differences,
    resample_regular_session_bars,
    validate_regular_session_minutes,
)

INDEX_ETFS = {
    "SPY": "ARCA",
    "QQQ": "NASDAQ",
    "DIA": "ARCA",
    "IWM": "ARCA",
}
OUTPUT_TIMEFRAMES = {1: "1m", 5: "5m", 15: "15m", 30: "30m"}
REQUEST_INTERVAL_SECONDS = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 8, 8))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 7))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=301)
    parser.add_argument("--chunk-root", type=Path, default=Path("data/raw/ibkr_chunks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--symbols", nargs="+", choices=tuple(INDEX_ETFS), default=list(INDEX_ETFS))
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-repair-sessions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end cannot precede --start")

    manifest: list[dict[str, object]] = []
    with IBKRHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    ) as client:
        for symbol in args.symbols:
            print(f"[{symbol}] resolving IBKR contract", flush=True)
            contract = client.qualify_us_stock(symbol, INDEX_ETFS[symbol])
            download_months(
                client,
                contract,
                start=args.start,
                end=args.end,
                chunk_root=args.chunk_root,
                max_retries=args.max_retries,
            )
            one_minute = load_chunks(symbol, args.chunk_root, start=args.start, end=args.end)
            one_minute, repair_sessions = repair_missing_sessions(
                client,
                contract,
                one_minute,
                start=args.start,
                end=args.end,
                chunk_root=args.chunk_root,
                max_retries=args.max_retries,
                max_repair_sessions=args.max_repair_sessions,
            )
            validate_regular_session_minutes(one_minute, start=args.start, end=args.end)

            outputs = write_timeframes(
                one_minute,
                symbol=symbol,
                output_root=args.output_root,
                metadata={
                    "provider": "IBKR",
                    "con_id": str(contract.conId),
                    "symbol": contract.symbol,
                    "exchange": contract.exchange,
                    "primary_exchange": contract.primaryExchange,
                    "currency": contract.currency,
                    "what_to_show": "TRADES",
                    "regular_trading_hours_only": "true",
                    "start_session": args.start.isoformat(),
                    "end_session": args.end.isoformat(),
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "repair_sessions": ",".join(session.isoformat() for session in repair_sessions),
                    "repair_exchange": contract.primaryExchange if repair_sessions else "",
                },
            )
            manifest.append(
                {
                    "symbol": symbol,
                    "con_id": contract.conId,
                    "primary_exchange": contract.primaryExchange,
                    "start": args.start.isoformat(),
                    "end": args.end.isoformat(),
                    "bars": {timeframe: count for timeframe, count, _ in outputs},
                    "files": {timeframe: str(path) for timeframe, _, path in outputs},
                }
            )
            print(
                f"[{symbol}] complete: "
                + ", ".join(f"{timeframe}={count:,}" for timeframe, count, _ in outputs),
                flush=True,
            )

    manifest_path = args.output_root / "us_index_etfs_manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}", flush=True)


def download_months(
    client: IBKRHistoricalClient,
    contract: object,
    *,
    start: date,
    end: date,
    chunk_root: Path,
    max_retries: int,
) -> None:
    """Download overlapping monthly windows; existing valid chunks are reused."""
    symbol = contract.symbol
    request_end = datetime.combine(end + timedelta(days=1), datetime_time.min, tzinfo=UTC)
    earliest_required = pd.Timestamp(start, tz=UTC)
    next_request_at = 0.0

    while request_end > earliest_required:
        chunk_path = chunk_root / f"symbol={symbol}" / f"end={request_end.date()}" / "bars.parquet"
        if chunk_path.is_file():
            frame = normalize_ohlcv(read_parquet_frame(chunk_path))
            print(
                f"[{symbol}] reuse {chunk_path.parent.name}: {len(frame):,} bars",
                flush=True,
            )
        else:
            wait_seconds = next_request_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            next_request_at = time.monotonic() + REQUEST_INTERVAL_SECONDS
            frame = request_with_retry(
                client,
                contract,
                end=request_end,
                max_retries=max_retries,
            )
            write_chunk(frame, chunk_path, symbol=symbol, request_end=request_end)
            print(
                f"[{symbol}] downloaded through {request_end.date()}: {len(frame):,} bars "
                f"({frame['timestamp'].iloc[0].date()} to {frame['timestamp'].iloc[-1].date()})",
                flush=True,
            )

        earliest = frame["timestamp"].iloc[0]
        if earliest <= earliest_required:
            break
        request_end = earliest.to_pydatetime()


def request_with_retry(
    client: IBKRHistoricalClient,
    contract: object,
    *,
    end: datetime,
    duration: str = "1 M",
    max_retries: int,
) -> tuple[pd.DataFrame, list[date]]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.request_minute_bars(contract, duration=duration, end=end)
        except Exception as error:
            last_error = error
            if attempt == max_retries:
                break
            delay = attempt * 10
            print(
                f"[{contract.symbol}] request ending {end.date()} failed "
                f"(attempt {attempt}/{max_retries}): {error}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"IBKR history failed for {contract.symbol} ending {end.date()} after "
        f"{max_retries} attempts"
    ) from last_error


def repair_missing_sessions(
    client: IBKRHistoricalClient,
    contract: object,
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
    chunk_root: Path,
    max_retries: int,
    max_repair_sessions: int,
) -> pd.DataFrame:
    """Re-request complete sessions when a monthly IBKR response has minute gaps."""
    missing, unexpected = regular_session_minute_differences(frame, start=start, end=end)
    if len(unexpected):
        raise ValueError(
            f"cannot repair data containing {len(unexpected)} unexpected regular-session timestamps"
        )
    if missing.empty:
        return frame, existing_repair_sessions(contract.symbol, chunk_root)

    previous_repair_sessions = existing_repair_sessions(contract.symbol, chunk_root)
    session_dates = sorted(set(missing.tz_convert("America/New_York").date))
    if len(session_dates) > max_repair_sessions:
        raise RuntimeError(
            f"refusing to repair {len(session_dates)} sessions; limit is {max_repair_sessions}"
        )

    print(
        f"[{contract.symbol}] repairing {len(missing):,} missing minutes across "
        f"{len(session_dates)} sessions",
        flush=True,
    )
    repair_contract = copy.copy(contract)
    repair_contract.exchange = contract.primaryExchange
    time.sleep(REQUEST_INTERVAL_SECONDS)
    for position, session_date in enumerate(session_dates):
        request_end = datetime.combine(
            session_date + timedelta(days=1),
            datetime_time.min,
            tzinfo=UTC,
        )
        repaired = request_with_retry(
            client,
            repair_contract,
            end=request_end,
            duration="1 D",
            max_retries=max_retries,
        )
        repair_path = (
            chunk_root
            / f"symbol={contract.symbol}"
            / f"repair={session_date.isoformat()}"
            / "bars.parquet"
        )
        write_chunk(
            repaired,
            repair_path,
            symbol=contract.symbol,
            request_end=request_end,
            request_duration="1 D",
            request_exchange=repair_contract.exchange,
        )
        print(
            f"[{contract.symbol}] repaired {session_date}: {len(repaired):,} bars",
            flush=True,
        )
        if position + 1 < len(session_dates):
            time.sleep(REQUEST_INTERVAL_SECONDS)

    repaired_frame = load_chunks(contract.symbol, chunk_root, start=start, end=end)
    return repaired_frame, sorted(set(previous_repair_sessions + session_dates))


def existing_repair_sessions(symbol: str, chunk_root: Path) -> list[date]:
    """Read already-applied repair session dates from the chunk directory."""
    repair_directories = (chunk_root / f"symbol={symbol}").glob("repair=*")
    return sorted(
        date.fromisoformat(directory.name.removeprefix("repair="))
        for directory in repair_directories
        if (directory / "bars.parquet").is_file()
    )


def write_manifest(entries: list[dict[str, object]], path: Path) -> None:
    """Merge completed symbols into the durable dataset manifest."""
    merged: dict[str, dict[str, object]] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"manifest must contain a JSON list: {path}")
        merged.update({str(entry["symbol"]): entry for entry in existing})
    merged.update({str(entry["symbol"]): entry for entry in entries})

    ordered = [merged[symbol] for symbol in INDEX_ETFS if symbol in merged]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def write_chunk(
    frame: pd.DataFrame,
    path: Path,
    *,
    symbol: str,
    request_end: datetime,
    request_duration: str = "1 M",
    request_exchange: str = "SMART",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(normalize_ohlcv(frame), preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"brooks_trader.schema": b"ohlcv-v1",
            b"brooks_trader.provider": b"IBKR",
            b"brooks_trader.symbol": symbol.encode(),
            b"brooks_trader.request_duration": request_duration.encode(),
            b"brooks_trader.request_end_utc": request_end.isoformat().encode(),
            b"brooks_trader.request_exchange": request_exchange.encode(),
        }
    )
    pq.write_table(table.replace_schema_metadata(metadata), path, compression="zstd")


def load_chunks(symbol: str, chunk_root: Path, *, start: date, end: date) -> pd.DataFrame:
    paths = sorted((chunk_root / f"symbol={symbol}").glob("*/bars.parquet"))
    if not paths:
        raise RuntimeError(f"no downloaded chunks found for {symbol}")
    combined = pd.concat((read_parquet_frame(path) for path in paths), ignore_index=True)
    normalized = normalize_ohlcv(combined.drop_duplicates(subset=["timestamp"], keep="last"))
    timestamps = normalized["timestamp"]
    session_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    selected = normalized.loc[(session_dates >= start) & (session_dates <= end)].reset_index(
        drop=True
    )
    if selected.empty:
        raise RuntimeError(f"downloaded chunks contain no requested data for {symbol}")
    return normalize_ohlcv(selected)


def write_timeframes(
    one_minute: pd.DataFrame,
    *,
    symbol: str,
    output_root: Path,
    metadata: dict[str, str],
) -> list[tuple[str, int, Path]]:
    outputs: list[tuple[str, int, Path]] = []
    for minutes, timeframe in OUTPUT_TIMEFRAMES.items():
        frame = one_minute if minutes == 1 else resample_regular_session_bars(one_minute, minutes)
        destination = output_root / f"symbol={symbol}" / f"timeframe={timeframe}" / "bars.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        parquet_metadata = dict(table.schema.metadata or {})
        parquet_metadata.update(
            {
                b"brooks_trader.schema": b"ohlcv-v1",
                b"brooks_trader.strategy_version": b"ibkr-historical-v1",
                b"brooks_trader.timeframe": timeframe.encode(),
                b"brooks_trader.derived_from": b"1m" if minutes != 1 else b"IBKR",
            }
        )
        for key, value in metadata.items():
            parquet_metadata[f"brooks_trader.source.{key}".encode()] = value.encode()
        pq.write_table(
            table.replace_schema_metadata(parquet_metadata),
            destination,
            compression="zstd",
        )
        outputs.append((timeframe, len(frame), destination))
    return outputs


if __name__ == "__main__":
    main()
