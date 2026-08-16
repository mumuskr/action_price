"""Parquet serialization for reproducible trade records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from brooks_trader.models import TradeRecord

TRADE_LOG_COLUMNS = (
    "trade_id",
    "symbol",
    "timeframe",
    "setup",
    "pattern_type",
    "direction",
    "market_regime",
    "pattern_score",
    "context_score",
    "signal_body_ratio",
    "signal_close_location",
    "ema_slope_ratio",
    "volatility_range_ratio",
    "entry_time",
    "entry_price",
    "quantity",
    "point_value",
    "slippage_ticks",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "initial_risk",
    "gross_pnl",
    "pnl",
    "commission",
    "pnl_r",
    "mfe",
    "mae",
    "mfe_r",
    "mae_r",
    "bars_held",
    "signal_bar_index",
    "market_state",
    "pattern_metadata",
    "entry_reason",
    "exit_reason",
    "strategy_version",
)


def trades_to_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    """Flatten immutable models into a stable, query-friendly schema."""
    records: list[dict[str, object]] = []
    for trade in trades:
        record = trade.model_dump()
        record["direction"] = trade.direction.value
        record["market_regime"] = trade.market_regime.value
        record["pattern_type"] = trade.pattern_type.value
        record["market_state"] = json.dumps(
            trade.market_state.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        record["pattern_metadata"] = json.dumps(
            trade.pattern_metadata,
            sort_keys=True,
            separators=(",", ":"),
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=TRADE_LOG_COLUMNS)


def write_trade_log(trades: list[TradeRecord], path: str | Path) -> Path:
    """Atomically write the completed trade log as Parquet."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    trades_to_frame(trades).to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination
