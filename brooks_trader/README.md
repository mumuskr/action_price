# Brooks Trader

Brooks Trader is a research system for translating subjective price-action concepts from
Al Brooks' *Trading Price Action* trilogy into explicit, configurable, testable
computational approximations.

The system keeps five concepts separate:

`Market Context -> Pattern -> Setup -> Signal -> Trade`

The trading engine, not an LLM, is the source of trading signals. LLM components are
restricted to knowledge extraction and explanation. Live execution will require human
confirmation by default.

## Current scope: Phase 10

The project now provides:

- the complete planned package layout;
- versioned YAML configuration;
- Pydantic domain models;
- strict local CSV/Parquet OHLCV loading;
- atomic Parquet dataset storage;
- an in-process DuckDB research query layer;
- causal bar features for body, tails, close location, doji, trend bar,
  inside/outside bar, HH/HL/LH/LL, and adjacent-bar overlap;
- configurable EMA and backward-only EMA slope calculations;
- an incremental Market Context Engine with EMA, structure, pressure, overlap,
  and breakout score components;
- seven configurable regime labels and confirmation-based Always In states;
- explicit H1/H2 and L1/L2 pullback state machines;
- traceable `PatternEvent` records and per-transition detector debug logs;
- a configurable Second Entry With Trend `SetupEngine` for H2/L2;
- separate setup evaluation and Trading Engine signal contracts;
- tick-aware stop entries, structural stops, and 1R/2R targets;
- a Trader's Equation that leaves probability and EV unknown until statistics exist;
- an event-driven bar-by-bar historical backtester with one position at a time;
- risk-based position sizing, gap/slippage/commission assumptions, and stop entries;
- conservative same-bar stop/target resolution and end-of-data liquidation;
- versioned trade logs with MFE/MAE, R multiples, and context snapshots;
- basic portfolio metrics and dedicated no-lookahead execution tests;
- empirical setup statistics by pattern, regime, signal quality, EMA slope, session,
  and volatility regime;
- a configurable minimum sample size before an observed win rate can be exposed as an
  empirical probability;
- local EPUB/Markdown extraction with paragraph-level source metadata;
- deterministic bounded knowledge chunks and a persistent local FAISS retrieval index;
- validated Brooks rule YAML with candidate/approved/rejected states and an audited human
  approval gate;
- a provider-neutral, read-only LLM explanation boundary around existing Trading Engine
  signals;
- structured explanation sections for context, setup, evidence, warnings, empirical
  statistics, Brooks references, risk/reward, and decision rationale;
- server-controlled prices, probability, strategy version, and RAG citations that an LLM
  response cannot replace; and
- failure-contained provider and response validation that preserves the original signal.
- a read-only Streamlit dashboard with Overview, Chart, Executed Signals, Trades, Setup
  Statistics, and Brooks Explanation views;
- causal EMA20 candlestick charts with saved entry, pattern, stop, and target markers; and
- explicit empty states when a market partition has no backtest or knowledge artifacts.

The historical `PaperBroker` is a deterministic research simulator. Live-stream paper
trading and external broker integration remain outside Phase 10. A detected pattern or
accepted setup is still not a broker order; only an accepted `StrategySignal` can be
submitted by the backtest layer.

## Installation

Python 3.12 or newer is required.

```bash
cd brooks_trader
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the local knowledge dependencies independently. UI dependencies are optional:

```bash
python -m pip install -e ".[knowledge,dev]"
```

For the Dashboard, install the app and research extras:

```bash
python -m pip install -e ".[app,research,dev]"
```

## Import local OHLCV

Input must contain `timestamp`, `open`, `high`, `low`, `close`, and `volume` columns.
Column names are matched case-insensitively. Timestamps are normalized to UTC, rows are
sorted chronologically, and invalid OHLC relationships or duplicate timestamps are rejected.

```bash
python scripts/import_data.py \
  --input data/raw/ES_5m.csv \
  --symbol ES \
  --timeframe 5m
```

The resulting dataset is stored at
`data/processed/symbol=ES/timeframe=5m/bars.parquet`.

For integer timestamps, provide an explicit unit such as `s`, `ms`, `us`, or `ns`:

```bash
python scripts/import_data.py \
  --input data/raw/ES_5m.csv \
  --symbol ES \
  --timeframe 5m \
  --timestamp-unit ms
```

## Download US stock data from IBKR

Start and log in to TWS or IB Gateway, enable its socket API, and keep the session
running. The downloader connects in read-only mode and does not synchronize account,
position, order, or execution data.

```bash
python scripts/download_ibkr.py \
  --symbol AAPL \
  --primary-exchange NASDAQ \
  --duration "1 D"
```

Paper TWS normally uses port `7497`; live TWS normally uses `7496`. The output is
`data/processed/symbol=AAPL/timeframe=1m/bars.parquet`. Timestamps are stored in UTC,
and IBKR request provenance is stored in Parquet metadata.

To build the ten-year US index ETF research dataset used by this project, keep Paper
TWS running and execute:

```bash
python scripts/download_us_indexes.py \
  --start 2016-08-08 \
  --end 2026-08-07
```

This downloads regular-session `TRADES` data for `SPY`, `QQQ`, `DIA`, and `IWM` at
one-minute resolution. It then validates every expected XNYS session minute and derives
the 5-, 15-, and 30-minute files locally. Monthly source chunks are retained under
`data/raw/ibkr_chunks/`, so an interrupted run resumes without downloading completed
windows again. Final files are stored under
`data/processed/symbol=<SYMBOL>/timeframe=<TIMEFRAME>/bars.parquet`, and the combined
inventory is `data/processed/us_index_etfs_manifest.json`.

## Query with DuckDB

```python
from brooks_trader.data import DuckDBQueryEngine

with DuckDBQueryEngine() as engine:
    engine.register_parquet("bars", "data/processed/**/*.parquet")
    result = engine.query("SELECT count(*) AS bar_count FROM bars")
```

## Calculate bar features

Feature thresholds and EMA parameters are loaded from `config/strategy.yaml`:

```python
import pandas as pd

from brooks_trader.features import calculate_bar_features, load_bar_feature_config

bars = pd.read_parquet("data/processed/symbol=SPY/timeframe=5m/bars.parquet")
config = load_bar_feature_config("config/strategy.yaml")
features = calculate_bar_features(bars, config=config)

print(features.tail())
```

All feature rows use only the current bar and earlier rows. The numeric definitions are
computational proxies for research and are not presented as formulas specified by Brooks.

## Detect market context

Market Context consumes the Phase 2 feature frame and returns one traceable `MarketState`
per bar. Batch and incremental calls share the same update path:

```python
from brooks_trader.market import MarketContextEngine, load_market_context_config

context_config, strategy_version = load_market_context_config("config/strategy.yaml")
engine = MarketContextEngine(context_config, strategy_version=strategy_version)
states = engine.detect(features)

print(states[-1].regime, states[-1].trend_score, states[-1].always_in)
```

The component scores and regime thresholds are configurable research proxies. They are
not claimed to be explicit formulas from Brooks' books, and the LLM is not involved in
their calculation.

## Detect H1/H2/L1/L2 patterns

The Phase 4 engine consumes synchronized bars, bar features, and market states. It emits
patterns only; it does not create orders or trade decisions:

```python
from brooks_trader.data.loader import bars_from_frame
from brooks_trader.patterns import (
    FirstSecondEntryPatternEngine,
    load_pattern_detector_config,
)

bars = bars_from_frame(raw_bars)
pattern_config, strategy_version = load_pattern_detector_config("config/strategy.yaml")
pattern_engine = FirstSecondEntryPatternEngine(
    pattern_config,
    strategy_version=strategy_version,
)
events = pattern_engine.detect(bars, features, states)

print(events[-1])
print(pattern_engine.debug_log[-1])
```

The computational proxy requires a pullback, a first break attempt, a new opposing leg,
and then a second break attempt. Consecutive same-direction bars alone cannot create H2
or L2. Pattern quality scores describe rule alignment and are not win probabilities.

## Evaluate Second Entry With Trend setups

Setup evaluation remains separate from pattern detection and broker execution:

```python
from brooks_trader.strategy import BrooksStrategy, SetupEngine, load_setup_engine_config

setup_config, strategy_version = load_setup_engine_config(
    "config/strategy.yaml",
    "config/markets.yaml",
    symbol="SPY",
)
setup_engine = SetupEngine(setup_config, strategy_version=strategy_version)
evaluation = setup_engine.evaluate(pattern, bars, features, states)

signal = BrooksStrategy(strategy_version=strategy_version).evaluate(evaluation)
```

Rejected evaluations preserve every rejection reason and do not contain a `TradeSetup`.
Accepted evaluations can produce a `StrategySignal`, but that signal has no quantity,
broker order state, or execution authority. Entry, stop, and target prices use the market's
configured tick size. Setup detection still leaves `probability_win` and expected value
as `None`; probabilities can only come from the separate, sample-qualified statistics
layer and are never guessed by an LLM.

## Run a historical backtest

Use a locally stored OHLCV dataset. The command replays every bar in order and writes an
auditable trade log and `setup_statistics.parquet` under the symbol/timeframe partition
in `data/backtests/`:

```bash
python scripts/run_backtest.py --symbol SPY --timeframe 5m
```

For a quick smoke test, limit the replay:

```bash
python scripts/run_backtest.py --symbol SPY --timeframe 5m --limit 10000
```

A signal produced after bar N closes is eligible from bar N+1. Stop entries account for
gaps and configured adverse slippage. If stop and target are both touched within a bar
without a decisive opening gap, the configured `adverse` policy selects the stop.
Capital, pending-order life, slippage, commission, and same-bar policy are configured in
`config/strategy.yaml`.

The statistics file contains descriptive `win_rate` for every observed group. The
`probability_win` column remains null until that exact group reaches
`statistics.minimum_probability_sample`. This prevents a small sample from silently
becoming a claimed trading probability.

## Build the Brooks knowledge base

Place EPUB or Markdown files that you are licensed to use under `books/raw/`. Source books
and generated artifacts are ignored by Git. Ingestion is entirely local:

```bash
python scripts/ingest_books.py
```

The parser writes paragraph-level records to `books/processed/processed_books.jsonl` and
builds a FAISS index under `books/processed/faiss/`. Query it with:

```bash
python scripts/query_knowledge.py "H2 second entry bull flag strong bull trend"
```

Every result carries a book/chapter/section/paragraph reference. The default
`brooks-hashing-v1` embedder is a deterministic local retrieval approximation, not a
trained semantic model and not a Brooks formula.

Files in `knowledge/patterns/` are documentation and review metadata, never executable
rule code. LLM-extracted rules are always forced to `candidate`. Approval requires a
verified source, reviewer identity, and timezone-aware review timestamp; production price
action remains implemented and tested in Python detectors and strategy engines.

## Explain an existing Trading Engine signal

The explainer is provider-neutral. Supply a small adapter for the model service selected by
the deployment; no model SDK is coupled to the Trading Engine:

```python
from brooks_trader.knowledge import FaissKnowledgeBase
from brooks_trader.llm import ExplanationRequest, LLMExplainer


class ConfiguredProvider:
    name = "configured-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        # Call the configured model here and return its JSON text response.
        return model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)


knowledge_base = FaissKnowledgeBase.load("books/processed/faiss")
explainer = LLMExplainer(
    provider=ConfiguredProvider(),
    knowledge_base=knowledge_base,
)
result = explainer.explain(
    ExplanationRequest(
        symbol="SPY",
        timeframe="5m",
        signal=strategy_signal,
        statistics=matching_setup_statistics,
        language="zh-CN",
    )
)
```

`strategy_signal` must already exist before explanation starts. The provider receives no
broker, portfolio, or order interface. It can author only four non-numeric narrative fields;
the application copies prices and reasons from `TradeSetup`, probability from the matching
sample-qualified `SetupStatistics`, and references from actual local FAISS results. Invalid
JSON, extra fields, numeric claims, retrieval errors, or provider errors produce a `FAILED`
explanation result while retaining the unchanged `StrategySignal`.

## Run the Dashboard

Start the read-only research UI from the project root:

```bash
.venv/bin/python scripts/run_dashboard.py
```

The launcher selects Arrow's system memory allocator before Streamlit starts. This is
required on macOS ARM because Arrow's default mimalloc backend can crash when Streamlit
serializes a large DataFrame from its worker thread.

The sidebar selects a canonical processed dataset under `data/processed/`. The result pages
load OHLCV and Parquet artifacts only; they never submit orders or change a saved backtest.
`Strategy Lab` is the explicit exception: it runs a selected historical experiment and
writes a new isolated artifact. `Overview` shows data coverage and completed-trade performance.
`Chart` calculates causal EMA20 features and marks saved trade entries and pattern bars.
`Executed Signals` lists signals that became completed trades; the current trade log does
not include rejected setups or unfilled/expired signals. `Trades` exposes the stored market
state and pattern metadata. `Setup Statistics` displays only empirical aggregates already
written by the statistics engine. `Brooks Explanation` retrieves passages from the local
FAISS index and shows the authoritative input boundary for the separate `LLMExplainer`.

### Run a strategy experiment

The `Strategy Lab` view exposes the executable strategy modules separately. Each module
shows its Brooks concept, the project's computational interpretation, implementation
status, and source file. `implemented` modules can be switched off before running an
experiment; `planned` modules are visible but disabled until their detector is connected
to the backtest pipeline.

Every dashboard run is saved under a unique experiment partition:

```text
data/backtests/symbol=SPY/timeframe=5m/experiment=<id>/
```

The partition contains `trades.parquet`, `setup_statistics.parquet`, and `metadata.json`
with the selected modules, configuration paths, metrics, and run counts. The sidebar's
`Backtest result` selector lets the read-only pages inspect the default artifact or any
saved experiment without overwriting earlier results.

The CLI uses the same module contract when a quick comparison is useful:

```bash
python scripts/run_backtest.py \
  --symbol SPY \
  --timeframe 5m \
  --disable-module ema_alignment_filter \
  --disable-module tight_trading_range_filter
```

Module definitions are cataloged in `src/brooks_trader/strategy/catalog.py`. A module's
description is intentionally separate from its implementation: book concepts, subjective
interpretations, and project-specific thresholds are not presented as the same thing.

## Quality checks

```bash
ruff check .
black --check .
pytest
```

## Design rule

Future numeric rules must be labeled as computational proxies, not presented as formulas
specified by Brooks. Thresholds belong in configuration and every emitted artifact must
remain traceable to its strategy version and source inputs.
