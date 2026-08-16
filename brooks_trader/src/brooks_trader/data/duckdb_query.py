"""Small read-oriented DuckDB facade for historical research."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb
import pandas as pd

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
QueryParameters = Sequence[Any] | Mapping[str, Any] | None


class DuckDBQueryEngine:
    """Manage a DuckDB connection and expose DataFrame query results."""

    def __init__(self, database: str | Path = ":memory:", *, read_only: bool = False) -> None:
        database_path = (
            str(database) if database == ":memory:" else str(Path(database).expanduser())
        )
        self._connection = duckdb.connect(database=database_path, read_only=read_only)
        self._closed = False

    def register_parquet(self, view_name: str, path_or_glob: str | Path) -> None:
        """Create or replace a view over one Parquet file or a glob."""
        identifier = _quote_identifier(view_name)
        path = str(Path(path_or_glob).expanduser())
        escaped_path = path.replace("'", "''")
        self._connection.execute(
            f"CREATE OR REPLACE VIEW {identifier} AS "
            f"SELECT * FROM read_parquet('{escaped_path}', hive_partitioning = true, "
            "union_by_name = true)"
        )

    def register_frame(self, view_name: str, frame: pd.DataFrame) -> None:
        """Register an in-memory DataFrame under a validated view name."""
        _quote_identifier(view_name)
        self._connection.register(view_name, frame)

    def query(self, sql: str, parameters: QueryParameters = None) -> pd.DataFrame:
        """Execute SQL and materialize its result as a pandas DataFrame."""
        if self._closed:
            raise RuntimeError("DuckDBQueryEngine is closed")
        if parameters is None:
            return self._connection.execute(sql).fetchdf()
        return self._connection.execute(sql, parameters).fetchdf()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _quote_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid DuckDB identifier: {value!r}")
    return f'"{value}"'
