from pathlib import Path

import pandas as pd
import pytest

from brooks_trader.features import calculate_bar_features, load_bar_feature_config
from brooks_trader.market import (
    AlwaysInTracker,
    MarketContextEngine,
    candidate_always_in,
    classify_regime,
    is_trading_range_score,
    load_market_context_config,
)
from brooks_trader.models import AlwaysInState, MarketRegime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_and_version():
    return load_market_context_config(PROJECT_ROOT / "config" / "strategy.yaml")


def test_classify_regime_uses_configured_boundaries(config_and_version) -> None:
    config, _ = config_and_version

    assert classify_regime(0.8, config) == MarketRegime.STRONG_BULL_TREND
    assert classify_regime(0.4, config) == MarketRegime.BULL_TREND
    assert classify_regime(0.2, config) == MarketRegime.WEAK_BULL
    assert classify_regime(0.1, config) == MarketRegime.TRADING_RANGE
    assert classify_regime(-0.2, config) == MarketRegime.WEAK_BEAR
    assert classify_regime(-0.4, config) == MarketRegime.BEAR_TREND
    assert classify_regime(-0.8, config) == MarketRegime.STRONG_BEAR_TREND
    assert is_trading_range_score(0.1, config)
    assert not is_trading_range_score(0.2, config)


def test_always_in_requires_confirmed_score_alignment(config_and_version) -> None:
    config, _ = config_and_version
    tracker = AlwaysInTracker(config)

    assert (
        candidate_always_in(
            trend_score=0.8,
            ema_score=0.2,
            structure_score=0.1,
            threshold=config.always_in_score_threshold,
        )
        == AlwaysInState.ALWAYS_IN_LONG
    )
    assert (
        tracker.update(trend_score=0.8, ema_score=0.2, structure_score=0.1) == AlwaysInState.NEUTRAL
    )
    assert (
        tracker.update(trend_score=0.8, ema_score=0.2, structure_score=0.1)
        == AlwaysInState.ALWAYS_IN_LONG
    )
    assert (
        tracker.update(trend_score=0.0, ema_score=0.0, structure_score=0.0)
        == AlwaysInState.ALWAYS_IN_LONG
    )
    assert (
        tracker.update(trend_score=-0.8, ema_score=-0.2, structure_score=-0.1)
        == AlwaysInState.ALWAYS_IN_LONG
    )
    assert (
        tracker.update(trend_score=-0.8, ema_score=-0.2, structure_score=-0.1)
        == AlwaysInState.ALWAYS_IN_SHORT
    )


def test_market_context_engine_returns_explainable_causal_states(config_and_version) -> None:
    context_config, version = config_and_version
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=12, freq="1min"),
            "open": [100.0 + index for index in range(12)],
            "high": [101.5 + index for index in range(12)],
            "low": [99.8 + index for index in range(12)],
            "close": [101.4 + index for index in range(12)],
            "volume": [100] * 12,
        }
    )
    bar_config = load_bar_feature_config(PROJECT_ROOT / "config" / "strategy.yaml")
    features = calculate_bar_features(bars, config=bar_config)

    states = MarketContextEngine(context_config, strategy_version=version).detect(features)

    assert len(states) == len(features)
    assert states[-1].regime in {MarketRegime.BULL_TREND, MarketRegime.STRONG_BULL_TREND}
    assert states[-1].always_in == AlwaysInState.ALWAYS_IN_LONG
    assert states[-1].trend_score > 0
    assert all(-1 <= state.trend_score <= 1 for state in states)
    assert all(-1 <= state.ema_score <= 1 for state in states)
    assert states[-1].strategy_version == "0.7.0-phase7"


def test_future_feature_change_cannot_alter_prior_market_states(config_and_version) -> None:
    context_config, version = config_and_version
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=16, freq="1min"),
            "open": [100.0 + index for index in range(16)],
            "high": [101.5 + index for index in range(16)],
            "low": [99.8 + index for index in range(16)],
            "close": [101.4 + index for index in range(16)],
            "volume": [100] * 16,
        }
    )
    changed = bars.copy()
    changed.loc[15, ["open", "high", "low", "close"]] = [90.0, 91.0, 80.0, 81.0]
    bar_config = load_bar_feature_config(PROJECT_ROOT / "config" / "strategy.yaml")
    engine = MarketContextEngine(context_config, strategy_version=version)

    original = engine.detect(calculate_bar_features(bars, config=bar_config))
    changed_states = engine.detect(calculate_bar_features(changed, config=bar_config))

    assert original[:15] == changed_states[:15]


def test_incremental_updates_match_batch_detection(config_and_version) -> None:
    context_config, version = config_and_version
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=10, freq="1min"),
            "open": [100.0 + index for index in range(10)],
            "high": [101.5 + index for index in range(10)],
            "low": [99.8 + index for index in range(10)],
            "close": [101.4 + index for index in range(10)],
            "volume": [100] * 10,
        }
    )
    bar_config = load_bar_feature_config(PROJECT_ROOT / "config" / "strategy.yaml")
    features = calculate_bar_features(bars, config=bar_config)
    batch_engine = MarketContextEngine(context_config, strategy_version=version)
    incremental_engine = MarketContextEngine(context_config, strategy_version=version)

    batch = batch_engine.detect(features)
    incremental = [incremental_engine.update(row) for _, row in features.iterrows()]

    assert incremental == batch


def test_market_context_engine_is_directionally_symmetric(config_and_version) -> None:
    context_config, version = config_and_version
    descending = list(reversed([100.0 + index for index in range(12)]))
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=12, freq="1min"),
            "open": [value + 1.4 for value in descending],
            "high": [value + 1.6 for value in descending],
            "low": [value - 0.1 for value in descending],
            "close": [value for value in descending],
            "volume": [100] * 12,
        }
    )
    bar_config = load_bar_feature_config(PROJECT_ROOT / "config" / "strategy.yaml")
    features = calculate_bar_features(bars, config=bar_config)

    states = MarketContextEngine(context_config, strategy_version=version).detect(features)

    assert states[-1].regime in {MarketRegime.BEAR_TREND, MarketRegime.STRONG_BEAR_TREND}
    assert states[-1].always_in == AlwaysInState.ALWAYS_IN_SHORT
    assert states[-1].trend_score < 0
