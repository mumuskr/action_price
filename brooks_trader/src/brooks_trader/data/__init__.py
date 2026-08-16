"""Local historical-data ingestion, storage, and research queries."""

from brooks_trader.data.duckdb_query import DuckDBQueryEngine
from brooks_trader.data.ibkr_history import download_us_stock_minute_bars
from brooks_trader.data.loader import (
    OHLCV_COLUMNS,
    OHLCVValidationError,
    bars_from_frame,
    load_ohlcv,
    normalize_ohlcv,
    read_parquet_frame,
)
from brooks_trader.data.parquet_store import ParquetStore

__all__ = [
    "OHLCV_COLUMNS",
    "DuckDBQueryEngine",
    "OHLCVValidationError",
    "ParquetStore",
    "bars_from_frame",
    "download_us_stock_minute_bars",
    "load_ohlcv",
    "normalize_ohlcv",
    "read_parquet_frame",
]
