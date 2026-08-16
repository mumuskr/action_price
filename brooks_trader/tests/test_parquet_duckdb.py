import pandas as pd
import pyarrow.parquet as pq

from brooks_trader.data import DuckDBQueryEngine, ParquetStore


def sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-02T09:30:00Z", "2026-01-02T09:31:00Z"],
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 100.0],
            "close": [100.5, 101.0],
            "volume": [10, 20],
        }
    )


def test_parquet_round_trip_and_metadata(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    expected = sample_bars()
    path = store.write_bars(
        expected,
        symbol="ES",
        timeframe="1m",
        strategy_version="test-v1",
    )

    assert path == tmp_path / "symbol=ES" / "timeframe=1m" / "bars.parquet"
    restored = store.read_bars(symbol="ES", timeframe="1m")
    assert restored[["open", "high", "low", "close", "volume"]].to_dict("list") == expected[
        ["open", "high", "low", "close", "volume"]
    ].to_dict("list")
    assert str(restored["timestamp"].dt.tz) == "UTC"
    metadata = pq.read_metadata(path).metadata
    assert metadata[b"brooks_trader.schema"] == b"ohlcv-v1"
    assert metadata[b"brooks_trader.strategy_version"] == b"test-v1"


def test_duckdb_queries_parquet_dataset(tmp_path) -> None:
    path = ParquetStore(tmp_path).write_bars(sample_bars(), symbol="ES", timeframe="1m")

    with DuckDBQueryEngine() as engine:
        engine.register_parquet("bars", path)
        result = engine.query("SELECT count(*) AS bar_count, max(close) AS maximum_close FROM bars")

    assert result.iloc[0].to_dict() == {"bar_count": 2.0, "maximum_close": 101.0}


def test_duckdb_parameterized_query() -> None:
    with DuckDBQueryEngine() as engine:
        result = engine.query("SELECT ?::INTEGER AS value", [42])

    assert result.iloc[0]["value"] == 42
