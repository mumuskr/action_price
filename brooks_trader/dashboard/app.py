"""Read-only Streamlit dashboard for historical Brooks Trader research artifacts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pyarrow as pa  # noqa: E402
import streamlit as st  # noqa: E402

from brooks_trader.backtest.runner import run_backtest_experiment  # noqa: E402
from brooks_trader.backtest.trade_logger import TRADE_LOG_COLUMNS  # noqa: E402
from brooks_trader.data import load_ohlcv, read_parquet_frame  # noqa: E402
from brooks_trader.features import (  # noqa: E402
    calculate_bar_features,
    load_bar_feature_config,
)
from brooks_trader.knowledge import FaissKnowledgeBase  # noqa: E402
from brooks_trader.statistics.setup_stats import STATISTICS_COLUMNS  # noqa: E402
from brooks_trader.strategy import (  # noqa: E402
    ModuleStatus,
    StrategyModuleSelection,
    strategy_module_catalog,
)

DATA_ROOT = PROJECT_ROOT / "data" / "processed"
BACKTEST_ROOT = PROJECT_ROOT / "data" / "backtests"
KNOWLEDGE_INDEX = PROJECT_ROOT / "books" / "processed" / "faiss"
STRATEGY_CONFIG = PROJECT_ROOT / "config" / "strategy.yaml"
MARKETS_CONFIG = PROJECT_ROOT / "config" / "markets.yaml"

PAGE_NAMES = (
    "Overview",
    "Chart",
    "Signals",
    "Trades",
    "Setup Statistics",
    "Brooks Explanation",
    "Strategy Lab",
)


@dataclass(frozen=True)
class DatasetRef:
    """One canonical symbol/timeframe market-data partition."""

    symbol: str
    timeframe: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.symbol} / {self.timeframe}"


@dataclass(frozen=True)
class BacktestArtifact:
    """One selectable baseline or isolated experiment result."""

    experiment_id: str
    label: str
    trade_path: Path
    statistics_path: Path
    metadata: dict[str, Any]


def discover_bar_datasets(root: str | Path = DATA_ROOT) -> tuple[DatasetRef, ...]:
    """Discover only canonical processed bar files, never raw downloader chunks."""
    source = Path(root).expanduser()
    datasets: list[DatasetRef] = []
    for path in sorted(source.glob("symbol=*/timeframe=*/bars.parquet")):
        symbol = path.parent.parent.name.removeprefix("symbol=")
        timeframe = path.parent.name.removeprefix("timeframe=")
        if symbol and timeframe:
            datasets.append(DatasetRef(symbol=symbol, timeframe=timeframe, path=path))
    return tuple(datasets)


def trade_log_path(
    symbol: str,
    timeframe: str,
    root: str | Path = BACKTEST_ROOT,
) -> Path:
    """Return the standard completed-trade log for one dataset."""
    return (
        Path(root).expanduser() / f"symbol={symbol}" / f"timeframe={timeframe}" / "trades.parquet"
    )


def statistics_path(
    symbol: str,
    timeframe: str,
    root: str | Path = BACKTEST_ROOT,
) -> Path:
    """Return the standard setup-statistics file for one dataset."""
    return (
        Path(root).expanduser()
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
        / "setup_statistics.parquet"
    )


def discover_backtest_artifacts(
    symbol: str,
    timeframe: str,
    root: str | Path = BACKTEST_ROOT,
) -> tuple[BacktestArtifact, ...]:
    """Discover the default result and all isolated strategy experiments."""
    partition = Path(root).expanduser() / f"symbol={symbol}" / f"timeframe={timeframe}"
    artifacts: list[BacktestArtifact] = []
    baseline_trade = partition / "trades.parquet"
    baseline_stats = partition / "setup_statistics.parquet"
    if baseline_trade.is_file() or baseline_stats.is_file():
        artifacts.append(
            BacktestArtifact(
                experiment_id="baseline",
                label="Default backtest",
                trade_path=baseline_trade,
                statistics_path=baseline_stats,
                metadata={"experiment_id": "baseline", "label": "Default backtest"},
            )
        )
    for metadata_path in sorted(partition.glob("experiment=*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        experiment_id = str(metadata.get("experiment_id") or metadata_path.parent.name)
        label = str(metadata.get("label") or experiment_id)
        artifacts.append(
            BacktestArtifact(
                experiment_id=experiment_id,
                label=f"{label} | {experiment_id}",
                trade_path=metadata_path.parent / str(metadata.get("trade_path", "trades.parquet")),
                statistics_path=metadata_path.parent
                / str(metadata.get("statistics_path", "setup_statistics.parquet")),
                metadata=metadata,
            )
        )
    return tuple(artifacts)


@st.cache_data(show_spinner=False)
def load_bars(path: str) -> pd.DataFrame:
    """Load and validate one canonical OHLCV file."""
    return load_ohlcv(path)


@st.cache_data(show_spinner=False)
def load_features(path: str, strategy_path: str) -> pd.DataFrame:
    """Calculate causal bar features for the selected OHLCV file."""
    bars = load_bars(path)
    config = load_bar_feature_config(strategy_path)
    return calculate_bar_features(bars, config=config)


@st.cache_data(show_spinner=False)
def load_trade_log(path: str) -> pd.DataFrame:
    """Load a completed trade log while preserving its stable schema."""
    source = Path(path)
    if not source.is_file():
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    frame = read_parquet_frame(source)
    for column in TRADE_LOG_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[:, list(TRADE_LOG_COLUMNS)].copy()
    for column in ("entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if "trade_id" in frame:
        frame = frame.sort_values(["entry_time", "trade_id"], kind="stable")
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_statistics(path: str) -> pd.DataFrame:
    """Load empirical statistics, returning an empty stable frame when unavailable."""
    source = Path(path)
    if not source.is_file():
        return pd.DataFrame(columns=STATISTICS_COLUMNS)
    frame = read_parquet_frame(source)
    for column in STATISTICS_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, list(STATISTICS_COLUMNS)].copy()


@st.cache_resource(show_spinner=False)
def load_knowledge_index(path: str) -> FaissKnowledgeBase:
    """Load the validated local FAISS index once per Streamlit process."""
    return FaissKnowledgeBase.load(path)


def main() -> None:
    """Render the dashboard application."""
    _require_system_arrow_memory_pool()
    st.set_page_config(
        page_title="Brooks Trader Research",
        page_icon="B",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()
    last_experiment = st.session_state.pop("last_experiment_message", None)
    if last_experiment:
        st.sidebar.success(last_experiment)
    datasets = discover_bar_datasets()
    st.sidebar.title("Brooks Trader")
    st.sidebar.caption("Historical research dashboard | read-only")
    page = st.sidebar.radio("View", PAGE_NAMES, index=0)
    if not datasets:
        st.title("Brooks Trader Research")
        st.warning("No processed OHLCV datasets were found.")
        st.code(
            "python scripts/import_data.py --input data/raw/SPY_5m.csv "
            "--symbol SPY --timeframe 5m",
            language="bash",
        )
        return

    selected_dataset = _select_dataset(datasets)
    selected_artifact = _select_backtest_artifact(selected_dataset)
    st.sidebar.caption(f"Result: {selected_artifact.label}")
    bars = load_bars(str(selected_dataset.path))
    trades = load_trade_log(str(selected_artifact.trade_path))
    statistics = load_statistics(str(selected_artifact.statistics_path))
    selected_trade = _select_trade(trades, page)

    if page == "Strategy Lab":
        render_strategy_lab(selected_dataset)
    elif page == "Overview":
        render_overview(selected_dataset, bars, trades, statistics)
    elif page == "Chart":
        features = load_features(str(selected_dataset.path), str(STRATEGY_CONFIG))
        render_chart(selected_dataset, bars, features, trades, selected_trade)
    elif page == "Signals":
        render_signals(selected_dataset, trades)
    elif page == "Trades":
        render_trades(selected_dataset, trades, selected_trade)
    elif page == "Setup Statistics":
        render_setup_statistics(selected_dataset, statistics)
    else:
        render_brooks_explanation(selected_dataset, trades, selected_trade)


def render_overview(
    dataset: DatasetRef,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Render data coverage and empirical performance without inventing unavailable values."""
    st.title("Overview")
    st.caption(f"{dataset.label} | UTC timestamps | source: {dataset.path}")
    start = bars["timestamp"].min()
    end = bars["timestamp"].max()
    completed = _completed_trades(trades)
    metrics = st.columns(5)
    metrics[0].metric("Bars", f"{len(bars):,}")
    metrics[1].metric("Last close", _number(bars["close"].iloc[-1]))
    metrics[2].metric("Completed trades", f"{len(completed):,}")
    metrics[3].metric("Win rate", _ratio(_win_rate(completed)))
    metrics[4].metric("Net PnL", _number(completed["pnl"].sum()) if len(completed) else "-")

    st.subheader("Data coverage")
    coverage = pd.DataFrame(
        {
            "Field": ["Start", "End", "Rows", "Average volume", "Statistics rows"],
            "Value": [
                _timestamp(start),
                _timestamp(end),
                f"{len(bars):,}",
                _number(bars["volume"].mean()),
                f"{len(statistics):,}",
            ],
        }
    )
    st.dataframe(coverage, hide_index=True, width="stretch")

    if completed.empty:
        st.info(
            "No completed backtest is available for this partition. "
            "Run the historical backtest to populate Signals, Trades, and statistics."
        )
        st.code(
            f"python scripts/run_backtest.py --symbol {dataset.symbol} "
            f"--timeframe {dataset.timeframe}",
            language="bash",
        )
    else:
        st.subheader("Performance snapshot")
        performance = completed[
            ["entry_time", "pnl", "pnl_r", "market_regime", "pattern_type"]
        ].copy()
        performance["cumulative_pnl"] = completed["pnl"].cumsum().to_numpy()
        st.line_chart(performance.set_index("entry_time")["cumulative_pnl"])
        st.dataframe(performance.tail(20), hide_index=True, width="stretch")


def render_strategy_lab(dataset: DatasetRef) -> None:
    """Render transparent module selection and run one isolated experiment."""
    st.title("Strategy Lab")
    st.caption(f"{dataset.label} | experiment artifacts are saved separately")

    fields = set(StrategyModuleSelection.model_fields)
    current_defaults = StrategyModuleSelection()
    modules = strategy_module_catalog()
    selectable_modules = [
        module
        for module in modules
        if module.id in fields and module.status != ModuleStatus.PLANNED
    ]
    selectable_ids = tuple(module.id for module in selectable_modules)
    for module in modules:
        widget_key = f"strategy_module_{module.id}"
        if widget_key in st.session_state:
            continue
        st.session_state[widget_key] = (
            getattr(current_defaults, module.id, module.default_enabled)
            if module.id in fields
            else False
        )

    select_all, clear_all, _spacer = st.columns([1, 1, 6])
    select_all.button(
        "全部勾选",
        width="stretch",
        on_click=_set_strategy_module_widgets,
        args=(selectable_ids, True),
    )
    clear_all.button(
        "取消全选",
        width="stretch",
        on_click=_set_strategy_module_widgets,
        args=(selectable_ids, False),
    )
    selected_count = sum(
        bool(st.session_state[f"strategy_module_{module_id}"]) for module_id in selectable_ids
    )
    st.caption(
        f"已选择 {selected_count} / {len(selectable_ids)} 个可运行模块; "
        "尚未接入回测的计划模块保持禁用。"
    )

    st.subheader("回测设置")
    experiment_column, replay_column = st.columns(2)
    experiment_label = experiment_column.text_input("实验名称", value="")
    replay_limit = int(
        replay_column.number_input(
            "回放 K 线数 (0 = 全部数据)",
            min_value=0,
            max_value=2_000_000,
            value=0,
            step=1_000,
        )
    )
    run_requested = st.button("运行回测", type="primary")
    progress_placeholder = st.empty()
    st.divider()

    selection_values: dict[str, bool] = {}
    for category in dict.fromkeys(module.category for module in modules):
        st.subheader(category)
        for module in [item for item in modules if item.category == category]:
            widget_key = f"strategy_module_{module.id}"
            enabled = st.checkbox(
                module.name,
                value=False,
                disabled=module.status == ModuleStatus.PLANNED,
                key=widget_key,
            )
            if module.id in fields:
                selection_values[module.id] = enabled
            status = module.status.value
            with st.expander(f"{module.name} | {status}"):
                st.write(module.description)
                st.caption(f"Book concept: {module.book_concept}")
                st.write(module.interpretation)
                source = PROJECT_ROOT / module.code_path
                st.caption(f"Code: {module.code_path}")
                if source.is_file():
                    st.code(source.read_text(encoding="utf-8"), language="python")
                else:
                    st.warning("Implementation file not found.")

    if not run_requested:
        return

    selection = StrategyModuleSelection.model_validate(selection_values)
    progress = progress_placeholder.progress(0, text="正在准备回测...")
    last_percent = -1

    def update_progress(completed: int, total: int) -> None:
        nonlocal last_percent
        percent = round((completed / total) * 90) if total else 90
        if percent == last_percent:
            return
        last_percent = percent
        progress.progress(
            percent,
            text=f"正在运行回测: {completed:,} / {total:,} 根 K 线",
        )

    try:
        _, _destination, metadata = run_backtest_experiment(
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
            strategy_path=STRATEGY_CONFIG,
            markets_path=MARKETS_CONFIG,
            output_root=BACKTEST_ROOT,
            module_selection=selection,
            limit=replay_limit or None,
            label=experiment_label,
            progress_callback=update_progress,
        )
        progress.progress(100, text="回测完成, 结果已保存。")
    except Exception as error:  # pragma: no cover - rendered by Streamlit
        st.error(f"Backtest failed: {error}")
        return

    st.session_state[f"selected_artifact:{dataset.label}"] = metadata["experiment_id"]
    st.session_state["last_experiment_message"] = (
        f"Experiment saved: {metadata['experiment_id']} ({metadata['trades']} trades)"
    )
    st.session_state.pop("strategy_module_run", None)
    st.rerun()


def _set_strategy_module_widgets(module_ids: tuple[str, ...], enabled: bool) -> None:
    """Set all executable strategy checkbox states before Streamlit renders them."""
    for module_id in module_ids:
        st.session_state[f"strategy_module_{module_id}"] = enabled


def render_chart(
    dataset: DatasetRef,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    trades: pd.DataFrame,
    selected_trade: pd.Series | None,
) -> None:
    """Render candles, EMA20, and traceable trade markers."""
    st.title("Chart")
    chart_bars = int(
        st.sidebar.number_input("Bars on chart", min_value=50, max_value=2000, value=300, step=50)
    )
    display_bars = bars.tail(chart_bars).copy()
    display_features = features.iloc[-len(display_bars) :]
    figure = build_price_chart(display_bars, display_features, bars, trades, selected_trade)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "Candles and EMA20 use causal OHLCV features. Markers are sourced from saved "
        "backtest trade records; the chart does not create signals."
    )
    if selected_trade is not None:
        _render_trade_prices(selected_trade)


def build_price_chart(
    display_bars: pd.DataFrame,
    display_features: pd.DataFrame,
    all_bars: pd.DataFrame,
    trades: pd.DataFrame,
    selected_trade: pd.Series | None = None,
) -> go.Figure:
    """Build a Plotly candlestick figure from already loaded data."""
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=display_bars["timestamp"],
            open=display_bars["open"],
            high=display_bars["high"],
            low=display_bars["low"],
            close=display_bars["close"],
            name="OHLC",
            increasing_line_color="#167c80",
            decreasing_line_color="#b14b4b",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=display_bars["timestamp"],
            y=display_features["ema20"],
            mode="lines",
            name="EMA20",
            line={"color": "#465a7a", "width": 1.5},
        )
    )
    _add_trade_markers(
        figure,
        all_bars,
        trades,
        start=display_bars["timestamp"].iloc[0],
        end=display_bars["timestamp"].iloc[-1],
    )
    if selected_trade is not None:
        _add_selected_trade_lines(figure, selected_trade)
    figure.update_layout(
        template="plotly_white",
        height=650,
        margin={"l": 15, "r": 15, "t": 30, "b": 15},
        xaxis={"rangeslider": {"visible": False}, "showgrid": False},
        yaxis={"title": "Price", "gridcolor": "#e8ebef"},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return figure


def render_signals(dataset: DatasetRef, trades: pd.DataFrame) -> None:
    """Render the persisted subset of signals that produced completed trades."""
    st.title("Executed Signals")
    st.caption(
        "The current backtest artifact stores signals that became completed trades. "
        "Rejected setups and unfilled or expired signals are not represented here. "
        "The Dashboard never emits a new signal."
    )
    completed = _completed_trades(trades)
    if completed.empty:
        st.info(f"No executed signals for {dataset.label}.")
        return
    table = completed[
        [
            "trade_id",
            "entry_time",
            "pattern_type",
            "direction",
            "market_regime",
            "entry_price",
            "stop_price",
            "target_price",
            "exit_reason",
            "pnl_r",
            "strategy_version",
        ]
    ].copy()
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download signals CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset.symbol}_{dataset.timeframe}_signals.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_trades(
    dataset: DatasetRef,
    trades: pd.DataFrame,
    selected_trade: pd.Series | None,
) -> None:
    """Render trade table and one selected trade's full point-in-time context."""
    st.title("Trades")
    completed = _completed_trades(trades)
    if completed.empty:
        st.info(f"No completed trades for {dataset.label}.")
        return
    st.dataframe(
        completed[
            [
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "pattern_type",
                "market_regime",
                "entry_price",
                "exit_price",
                "pnl",
                "pnl_r",
                "mfe_r",
                "mae_r",
                "bars_held",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    if selected_trade is None:
        return
    st.subheader(f"Trade detail: {selected_trade['trade_id']}")
    detail_columns = st.columns(3)
    detail_columns[0].metric("Entry", _number(selected_trade["entry_price"]))
    detail_columns[1].metric("Exit", _number(selected_trade["exit_price"]))
    detail_columns[2].metric("Pnl (R)", _number(selected_trade["pnl_r"]))
    left, right = st.columns(2)
    with left:
        st.markdown("**Trade facts**")
        st.dataframe(_trade_facts(selected_trade), hide_index=True, width="stretch")
        st.markdown("**Market state**")
        st.json(_json_value(selected_trade.get("market_state")))
    with right:
        st.markdown("**Pattern metadata**")
        st.json(_json_value(selected_trade.get("pattern_metadata")))
        st.markdown("**Reasons**")
        st.write(selected_trade.get("entry_reason", "-"))
        st.markdown("**Exit reason**")
        st.write(selected_trade.get("exit_reason", "-"))


def render_setup_statistics(dataset: DatasetRef, statistics: pd.DataFrame) -> None:
    """Render empirical setup aggregates exactly as stored by the backtester."""
    st.title("Setup Statistics")
    st.caption(
        "All rates and expectancy values below come from completed trades. "
        "A missing probability is not estimated by the Dashboard."
    )
    if statistics.empty:
        st.info(f"No setup statistics for {dataset.label}.")
        return
    patterns = [
        "All",
        *sorted(statistics["pattern_type"].dropna().astype(str).unique().tolist()),
    ]
    regimes = [
        "All",
        *sorted(statistics["market_regime"].dropna().astype(str).unique().tolist()),
    ]
    pattern = st.selectbox("Pattern", patterns)
    regime = st.selectbox("Market regime", regimes)
    filtered = statistics.copy()
    if pattern != "All":
        filtered = filtered[filtered["pattern_type"].astype(str) == pattern]
    if regime != "All":
        filtered = filtered[filtered["market_regime"].astype(str) == regime]
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.download_button(
        "Download statistics CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset.symbol}_{dataset.timeframe}_setup_statistics.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    if not filtered.empty and "expectancy_r" in filtered:
        chart_data = filtered.dropna(subset=["expectancy_r"]).copy()
        if not chart_data.empty:
            chart_data["group"] = chart_data.apply(_statistics_group_label, axis=1)
            figure = go.Figure(
                go.Bar(
                    x=chart_data["group"],
                    y=chart_data["expectancy_r"],
                    marker_color=[
                        "#167c80" if value >= 0 else "#b14b4b"
                        for value in chart_data["expectancy_r"]
                    ],
                )
            )
            figure.update_layout(
                template="plotly_white",
                height=360,
                yaxis_title="Expectancy (R)",
                xaxis_title="Statistics group",
                margin={"l": 15, "r": 15, "t": 30, "b": 90},
            )
            st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_brooks_explanation(
    dataset: DatasetRef,
    trades: pd.DataFrame,
    selected_trade: pd.Series | None,
) -> None:
    """Render authoritative context and local Brooks RAG evidence."""
    st.title("Brooks Explanation")
    st.caption(
        "This page is read-only. RAG evidence is local and source-backed; an LLM provider "
        "is not connected by the Dashboard."
    )
    if selected_trade is not None:
        st.subheader("Authoritative explanation input")
        st.json(_explanation_input(dataset, selected_trade))
        query_default = _trade_query(selected_trade)
    else:
        query_default = "H2 second entry bull flag strong bull trend"
        st.info("Select a saved trade in the sidebar to inspect its context and evidence.")
    query = st.text_input("Brooks evidence query", value=query_default)
    top_k = int(st.number_input("Retrieved passages", min_value=1, max_value=10, value=5))
    if not KNOWLEDGE_INDEX.is_dir():
        st.warning("Local FAISS knowledge index is not available.")
        st.code("python scripts/ingest_books.py", language="bash")
        return
    try:
        knowledge_base = load_knowledge_index(str(KNOWLEDGE_INDEX))
        results = knowledge_base.search(query, top_k=top_k)
    except (OSError, ValueError) as error:
        st.error(f"Knowledge retrieval failed: {error}")
        return
    st.subheader(f"Brooks references ({len(results)})")
    if not results:
        st.info("No matching source passages were retrieved.")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        with st.expander(f"{rank}. {chunk.source_reference} | score {result.score:.4f}"):
            st.write(chunk.text)
            st.caption(f"Chunk ID: {chunk.chunk_id}")
    with st.expander("LLM boundary"):
        st.write(
            "The provider-neutral LLMExplainer can consume this signal, market context, "
            "statistics, and these references. It cannot create a BUY or SELL signal, "
            "change prices, change quantity, or submit a broker order."
        )


def _select_dataset(datasets: tuple[DatasetRef, ...]) -> DatasetRef:
    labels = [dataset.label for dataset in datasets]
    selected_label = st.sidebar.selectbox("Dataset", labels)
    return datasets[labels.index(selected_label)]


def _select_backtest_artifact(dataset: DatasetRef) -> BacktestArtifact:
    artifacts = discover_backtest_artifacts(dataset.symbol, dataset.timeframe)
    if not artifacts:
        return BacktestArtifact(
            experiment_id="none",
            label="No backtest",
            trade_path=trade_log_path(dataset.symbol, dataset.timeframe),
            statistics_path=statistics_path(dataset.symbol, dataset.timeframe),
            metadata={},
        )
    labels = [artifact.label for artifact in artifacts]
    state_key = f"selected_artifact:{dataset.label}"
    selected_id = st.session_state.get(state_key)
    index = next(
        (
            position
            for position, artifact in enumerate(artifacts)
            if artifact.experiment_id == selected_id
        ),
        0,
    )
    widget_key = f"backtest_result_{dataset.symbol}_{dataset.timeframe}"
    if selected_id is not None and index < len(labels):
        st.session_state[widget_key] = labels[index]
    selected_label = st.sidebar.selectbox(
        "Backtest result",
        labels,
        index=index,
        key=widget_key,
    )
    selected = artifacts[labels.index(selected_label)]
    st.session_state[state_key] = selected.experiment_id
    return selected


def _select_trade(trades: pd.DataFrame, page: str) -> pd.Series | None:
    completed = _completed_trades(trades)
    if completed.empty or page not in {"Chart", "Trades", "Brooks Explanation"}:
        return None
    options = completed["trade_id"].astype(str).tolist()
    selected_id = st.sidebar.selectbox("Selected trade", options, key="selected_trade_id")
    return completed.loc[completed["trade_id"].astype(str) == selected_id].iloc[0]


def _completed_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades[trades["exit_time"].notna() & trades["exit_price"].notna()].copy()


def _win_rate(trades: pd.DataFrame) -> float | None:
    if trades.empty:
        return None
    return float((pd.to_numeric(trades["pnl"], errors="coerce") > 0).mean())


def _add_trade_markers(
    figure: go.Figure,
    all_bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    start: Any,
    end: Any,
) -> None:
    """Add entry and pattern markers from saved trade rows."""
    completed = _completed_trades(trades)
    if completed.empty:
        return
    signal_points: list[dict[str, Any]] = []
    entry_points: list[dict[str, Any]] = []
    for _, trade in completed.iterrows():
        signal_index = _int_or_none(trade.get("signal_bar_index"))
        if signal_index is not None and 0 <= signal_index < len(all_bars):
            bar = all_bars.iloc[signal_index]
            if start <= bar["timestamp"] <= end:
                signal_points.append(
                    {
                        "x": bar["timestamp"],
                        "y": bar["high"] if str(trade["direction"]) == "LONG" else bar["low"],
                        "text": f"{trade['pattern_type']} | {trade['trade_id']}",
                    }
                )
        if (
            pd.notna(trade.get("entry_time"))
            and pd.notna(trade.get("entry_price"))
            and start <= trade["entry_time"] <= end
        ):
            entry_points.append(
                {
                    "x": trade["entry_time"],
                    "y": trade["entry_price"],
                    "text": f"Entry | {trade['trade_id']}",
                }
            )
    if signal_points:
        figure.add_trace(
            go.Scatter(
                x=[point["x"] for point in signal_points],
                y=[point["y"] for point in signal_points],
                text=[point["text"] for point in signal_points],
                mode="markers+text",
                textposition="top center",
                marker={"symbol": "diamond", "size": 8, "color": "#a06a00"},
                name="Pattern",
            )
        )
    if entry_points:
        figure.add_trace(
            go.Scatter(
                x=[point["x"] for point in entry_points],
                y=[point["y"] for point in entry_points],
                text=[point["text"] for point in entry_points],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 9, "color": "#167c80"},
                name="Entry",
            )
        )


def _add_selected_trade_lines(figure: go.Figure, trade: pd.Series) -> None:
    for name, field, color in (
        ("Entry", "entry_price", "#167c80"),
        ("Stop", "stop_price", "#b14b4b"),
        ("Target", "target_price", "#a06a00"),
    ):
        value = pd.to_numeric(trade.get(field), errors="coerce")
        if pd.notna(value):
            figure.add_hline(
                y=float(value),
                line_dash="dot",
                line_color=color,
                annotation_text=name,
                annotation_position="top left",
            )


def _render_trade_prices(trade: pd.Series) -> None:
    prices = pd.DataFrame(
        {
            "Level": ["Entry", "Stop", "Target", "Exit"],
            "Price": [
                trade.get("entry_price"),
                trade.get("stop_price"),
                trade.get("target_price"),
                trade.get("exit_price"),
            ],
        }
    )
    st.dataframe(prices, hide_index=True, width="stretch")


def _trade_facts(trade: pd.Series) -> pd.DataFrame:
    fields = (
        ("Trade ID", "trade_id"),
        ("Setup", "setup"),
        ("Pattern", "pattern_type"),
        ("Direction", "direction"),
        ("Regime", "market_regime"),
        ("Entry time", "entry_time"),
        ("Exit time", "exit_time"),
        ("Strategy version", "strategy_version"),
        ("Bars held", "bars_held"),
    )
    return pd.DataFrame(
        {
            "Field": [label for label, _ in fields],
            "Value": [str(trade.get(field, "-")) for _, field in fields],
        }
    )


def _explanation_input(dataset: DatasetRef, trade: pd.Series) -> dict[str, Any]:
    return {
        "symbol": dataset.symbol,
        "timeframe": dataset.timeframe,
        "trade_id": trade.get("trade_id"),
        "setup": trade.get("setup"),
        "pattern_type": trade.get("pattern_type"),
        "direction": trade.get("direction"),
        "market_regime": trade.get("market_regime"),
        "entry": trade.get("entry_price"),
        "stop": trade.get("stop_price"),
        "target": trade.get("target_price"),
        "pnl_r": trade.get("pnl_r"),
        "strategy_version": trade.get("strategy_version"),
        "market_state": _json_value(trade.get("market_state")),
        "pattern_metadata": _json_value(trade.get("pattern_metadata")),
    }


def _trade_query(trade: pd.Series) -> str:
    pattern = str(trade.get("pattern_type", "H2"))
    direction = str(trade.get("direction", "LONG"))
    context = "bull flag bull trend" if direction == "LONG" else "bear flag bear trend"
    return f"{pattern} second entry {context}"


def _statistics_group_label(row: pd.Series) -> str:
    values = [row.get("scope"), row.get("pattern_type"), row.get("market_regime")]
    return " / ".join(str(value) for value in values if pd.notna(value) and str(value) != "")


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}


def _int_or_none(value: Any) -> int | None:
    converted = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(converted) else int(converted)


def _number(value: Any) -> str:
    converted = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(converted) else f"{float(converted):,.4f}"


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _timestamp(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        [data-testid="stSidebar"] { border-right: 1px solid #dfe4ea; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_system_arrow_memory_pool() -> None:
    """Reject an unsafe launch before Streamlit serializes data on a worker thread."""
    backend = pa.default_memory_pool().backend_name
    if backend != "system":
        raise RuntimeError(
            "Dashboard requires the Arrow system memory pool; start it with "
            "python scripts/run_dashboard.py"
        )


if __name__ == "__main__":
    main()
