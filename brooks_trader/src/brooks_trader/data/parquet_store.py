"""Atomic Parquet persistence for canonical historical bars."""

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brooks_trader.data.loader import TimestampUnit, normalize_ohlcv, read_parquet_frame

_PARTITION_VALUE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ParquetStore:
    """Store one canonical bar file per symbol and timeframe partition."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def dataset_path(self, symbol: str, timeframe: str) -> Path:
        """Return a safe Hive-style partition path."""
        safe_symbol = _validate_partition_value("symbol", symbol)
        safe_timeframe = _validate_partition_value("timeframe", timeframe)
        return self.root / f"symbol={safe_symbol}" / f"timeframe={safe_timeframe}" / "bars.parquet"

    def write_bars(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        timestamp_unit: TimestampUnit | None = None,
        strategy_version: str = "data-schema-v1",
        source_metadata: Mapping[str, str] | None = None,
    ) -> Path:
        """Validate and atomically replace a symbol/timeframe Parquet dataset."""
        normalized = normalize_ohlcv(frame, timestamp_unit=timestamp_unit)
        destination = self.dataset_path(symbol, timeframe)
        destination.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(normalized, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"brooks_trader.schema": b"ohlcv-v1",
                b"brooks_trader.symbol": symbol.encode("utf-8"),
                b"brooks_trader.timeframe": timeframe.encode("utf-8"),
                b"brooks_trader.strategy_version": strategy_version.encode("utf-8"),
            }
        )
        for key, value in (source_metadata or {}).items():
            metadata[f"brooks_trader.source.{key}".encode()] = value.encode()
        table = table.replace_schema_metadata(metadata)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".bars-",
                suffix=".parquet.tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            pq.write_table(table, temporary_path, compression="zstd")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination

    def read_bars(self, *, symbol: str, timeframe: str) -> pd.DataFrame:
        """Read and revalidate a stored symbol/timeframe dataset."""
        path = self.dataset_path(symbol, timeframe)
        if not path.is_file():
            raise FileNotFoundError(f"Parquet dataset does not exist: {path}")
        return normalize_ohlcv(read_parquet_frame(path))


def _validate_partition_value(name: str, value: str) -> str:
    stripped = value.strip()
    if not stripped or _PARTITION_VALUE.fullmatch(stripped) is None:
        raise ValueError(
            f"{name} must contain only letters, numbers, underscores, dots, and hyphens"
        )
    return stripped
