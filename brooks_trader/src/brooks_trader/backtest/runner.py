"""Reusable backtest experiment runner for the CLI and dashboard."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from brooks_trader.backtest.engine import BacktestEngine, BacktestResult
from brooks_trader.backtest.trade_logger import write_trade_log
from brooks_trader.data import load_ohlcv
from brooks_trader.statistics import (
    calculate_setup_statistics,
    load_setup_statistics_config,
    write_setup_statistics,
)
from brooks_trader.strategy.catalog import StrategyModuleSelection


def run_backtest_experiment(
    *,
    symbol: str,
    timeframe: str,
    strategy_path: str | Path,
    markets_path: str | Path,
    output_root: str | Path,
    module_selection: StrategyModuleSelection | None = None,
    limit: int | None = None,
    label: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[BacktestResult, Path, dict[str, Any]]:
    """Run one isolated experiment and persist all artifacts and provenance."""
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = timeframe.strip()
    if not normalized_symbol or not normalized_timeframe:
        raise ValueError("symbol and timeframe must be non-empty")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one")

    source = (
        Path("data/processed")
        / f"symbol={normalized_symbol}"
        / f"timeframe={normalized_timeframe}"
        / "bars.parquet"
    )
    frame = load_ohlcv(source)
    if limit is not None:
        frame = frame.iloc[:limit].copy()

    selection = module_selection or StrategyModuleSelection()
    engine = BacktestEngine.from_config(
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        strategy_path=strategy_path,
        markets_path=markets_path,
        module_overrides=selection,
    )
    result = engine.run(frame, progress_callback=progress_callback)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    selection_slug = _selection_slug(selection)
    experiment_id = f"{timestamp}_{selection_slug}_{uuid4().hex[:8]}"
    if label and label.strip():
        experiment_id = f"{_slug(label)}_{experiment_id}"
    destination = (
        Path(output_root).expanduser()
        / f"symbol={normalized_symbol}"
        / f"timeframe={normalized_timeframe}"
        / f"experiment={experiment_id}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    trade_path = write_trade_log(list(result.trades), destination / "trades.parquet")
    statistics = calculate_setup_statistics(
        result.trades,
        config=load_setup_statistics_config(strategy_path),
    )
    statistics_path = write_setup_statistics(statistics, destination / "setup_statistics.parquet")
    metadata: dict[str, Any] = {
        "experiment_id": experiment_id,
        "label": label.strip() if label and label.strip() else None,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "strategy_version": result.strategy_version,
        "strategy_path": str(Path(strategy_path)),
        "markets_path": str(Path(markets_path)),
        "module_selection": selection.model_dump(),
        "enabled_modules": list(selection.enabled_ids()),
        "bars": len(frame),
        "patterns": len(result.patterns),
        "signals": len(result.signals),
        "trades": len(result.trades),
        "metrics": asdict(result.metrics),
        "ending_cash": result.ending_cash,
        "trade_path": str(trade_path.name),
        "statistics_path": str(statistics_path.name),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return result, destination, metadata


def _selection_slug(selection: StrategyModuleSelection) -> str:
    enabled = selection.enabled_ids()
    return _slug("-".join(enabled) if enabled else "no-modules")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized[:100] or "experiment"
