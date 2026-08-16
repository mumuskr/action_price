"""Read-only US stock history retrieval through a running IBKR TWS session."""

from datetime import UTC, datetime
from types import TracebackType
from typing import Self

import pandas as pd
from ib_async import IB, BarData, StartupFetchNONE, Stock

from brooks_trader.data.loader import normalize_ohlcv

_INFORMATIONAL_CODES = {2104, 2106, 2158}


class IBKRHistoricalClient:
    """A reusable read-only IBKR connection for historical bars only."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 201,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = IB()
        self._errors: list[tuple[int, str]] = []
        self._ib.errorEvent += self._record_error

    def connect(self) -> Self:
        self._ib.connect(
            host=self.host,
            port=self.port,
            clientId=self.client_id,
            timeout=10,
            readonly=True,
            raiseSyncErrors=True,
            fetchFields=StartupFetchNONE,
        )
        return self

    def qualify_us_stock(self, symbol: str, primary_exchange: str) -> Stock:
        normalized_symbol = symbol.strip().upper()
        normalized_exchange = primary_exchange.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol cannot be empty")
        if not normalized_exchange:
            raise ValueError("primary_exchange cannot be empty")

        contract = Stock(
            normalized_symbol,
            "SMART",
            "USD",
            primaryExchange=normalized_exchange,
        )
        qualified = self._ib.qualifyContracts(contract)
        if len(qualified) != 1:
            raise RuntimeError(
                f"IBKR did not resolve exactly one contract for {normalized_symbol}: {qualified!r}"
            )
        return qualified[0]

    def request_minute_bars(
        self,
        contract: Stock,
        *,
        duration: str,
        end: datetime | None,
    ) -> pd.DataFrame:
        if end is not None and (end.tzinfo is None or end.utcoffset() is None):
            raise ValueError("end must be timezone-aware")
        self._errors.clear()
        bars = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime=end.astimezone(UTC) if end is not None else "",
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            timeout=90,
        )
        if not bars:
            details = "; ".join(f"IBKR {code}: {message}" for code, message in self._errors)
            raise RuntimeError(details or f"IBKR returned no historical bars for {contract.symbol}")
        return ibkr_bars_to_frame(bars)

    def disconnect(self) -> None:
        self._ib.disconnect()

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    def _record_error(
        self,
        _request_id: int,
        error_code: int,
        error_message: str,
        _contract: object,
    ) -> None:
        if error_code not in _INFORMATIONAL_CODES:
            self._errors.append((error_code, error_message))


def download_us_stock_minute_bars(
    symbol: str,
    *,
    primary_exchange: str,
    duration: str = "1 D",
    end: datetime | None = None,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 201,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download regular-session, unadjusted one-minute trades from IBKR.

    The connection is read-only and explicitly skips account, position, order,
    execution, and account-update synchronization.
    """
    with IBKRHistoricalClient(host=host, port=port, client_id=client_id) as client:
        resolved = client.qualify_us_stock(symbol, primary_exchange)
        frame = client.request_minute_bars(resolved, duration=duration, end=end)
        metadata = {
            "provider": "IBKR",
            "con_id": str(resolved.conId),
            "symbol": resolved.symbol,
            "exchange": resolved.exchange,
            "primary_exchange": resolved.primaryExchange,
            "currency": resolved.currency,
            "bar_size": "1 min",
            "what_to_show": "TRADES",
            "regular_trading_hours_only": "true",
            "duration": duration,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
        }
        return frame, metadata


def ibkr_bars_to_frame(bars: list[BarData]) -> pd.DataFrame:
    """Convert IBKR intraday bars to the canonical, strictly validated schema."""
    if not bars:
        raise ValueError("IBKR bar collection cannot be empty")
    frame = pd.DataFrame.from_records(
        {
            "timestamp": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    )
    return normalize_ohlcv(frame)
