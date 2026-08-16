from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from dashboard.app import (
    _require_system_arrow_memory_pool,
    build_price_chart,
    discover_bar_datasets,
    load_statistics,
    load_trade_log,
    statistics_path,
    trade_log_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_rejects_unsafe_arrow_memory_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MimallocPool:
        backend_name = "mimalloc"

    monkeypatch.setattr(pa, "default_memory_pool", MimallocPool)

    with pytest.raises(RuntimeError, match="system memory pool"):
        _require_system_arrow_memory_pool()


def test_dashboard_bootstraps_source_import_path() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["ARROW_DEFAULT_MEMORY_POOL"] = "system"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dashboard.app; import brooks_trader; print(brooks_trader.__version__)",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.10.1"


def test_discover_bar_datasets_finds_only_canonical_partitions(tmp_path: Path) -> None:
    canonical = tmp_path / "symbol=SPY" / "timeframe=5m" / "bars.parquet"
    canonical.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T14:30:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
        }
    ).to_parquet(canonical, index=False)
    (tmp_path / "raw" / "symbol=SPY" / "end=2026-01-01").mkdir(parents=True)

    datasets = discover_bar_datasets(tmp_path)

    assert [(item.symbol, item.timeframe) for item in datasets] == [("SPY", "5m")]
    assert datasets[0].label == "SPY / 5m"


def test_standard_artifact_paths_are_partitioned() -> None:
    assert trade_log_path("SPY", "5m", "/tmp/backtests") == Path(
        "/tmp/backtests/symbol=SPY/timeframe=5m/trades.parquet"
    )
    assert statistics_path("SPY", "5m", "/tmp/backtests") == Path(
        "/tmp/backtests/symbol=SPY/timeframe=5m/setup_statistics.parquet"
    )


def test_missing_artifacts_return_stable_empty_frames(tmp_path: Path) -> None:
    trades = load_trade_log(str(tmp_path / "missing-trades.parquet"))
    statistics = load_statistics(str(tmp_path / "missing-statistics.parquet"))

    assert trades.empty
    assert "strategy_version" in trades.columns
    assert statistics.empty
    assert "probability_win" in statistics.columns


def test_price_chart_contains_candles_and_ema_without_trade_data() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "open": [100.0, 101.0, 100.5],
            "high": [102.0, 102.0, 101.5],
            "low": [99.0, 100.0, 99.5],
            "close": [101.0, 100.5, 101.25],
            "volume": [100, 110, 120],
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "ema20": [100.0, 100.1, 100.2],
        }
    )

    figure = build_price_chart(bars, features, bars, pd.DataFrame())

    assert len(figure.data) == 2
    assert figure.data[0].type == "candlestick"
    assert figure.data[1].name == "EMA20"
