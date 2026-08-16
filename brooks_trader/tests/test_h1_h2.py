from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from brooks_trader.data.loader import bars_from_frame
from brooks_trader.features import calculate_bar_features, load_bar_feature_config
from brooks_trader.models import (
    AlwaysInState,
    MarketRegime,
    MarketState,
    PatternType,
)
from brooks_trader.patterns import (
    FirstSecondEntryPatternEngine,
    H1H2Detector,
    L1L2Detector,
    load_pattern_detector_config,
)
from brooks_trader.patterns.base import PatternDetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = PROJECT_ROOT / "config" / "strategy.yaml"


def bull_contexts(features: pd.DataFrame) -> list[MarketState]:
    return [
        MarketState(
            timestamp=row.timestamp,
            bar_index=int(row.bar_index),
            regime=MarketRegime.BULL_TREND,
            always_in=AlwaysInState.ALWAYS_IN_LONG,
            trend_score=0.6,
            ema_score=0.5,
            structure_score=0.4,
            pressure_score=0.3,
            overlap_score=0.2,
            breakout_score=0.2,
            strategy_version="0.7.0-phase7",
        )
        for row in features.itertuples(index=False)
    ]


def bear_contexts(features: pd.DataFrame) -> list[MarketState]:
    return [
        MarketState(
            timestamp=row.timestamp,
            bar_index=int(row.bar_index),
            regime=MarketRegime.BEAR_TREND,
            always_in=AlwaysInState.ALWAYS_IN_SHORT,
            trend_score=-0.6,
            ema_score=-0.5,
            structure_score=-0.4,
            pressure_score=-0.3,
            overlap_score=-0.2,
            breakout_score=-0.2,
            strategy_version="0.7.0-phase7",
        )
        for row in features.itertuples(index=False)
    ]


def feature_inputs(frame: pd.DataFrame):
    bar_config = load_bar_feature_config(STRATEGY_PATH)
    normalized_bars = bars_from_frame(frame)
    features = calculate_bar_features(frame, config=bar_config)
    return normalized_bars, features


def test_h1_h2_state_machine_emits_first_and_second_attempts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=5, freq="1min"),
            "open": [100.0, 101.0, 100.2, 101.5, 100.0],
            "high": [102.0, 101.5, 101.8, 101.6, 102.0],
            "low": [99.0, 100.0, 100.1, 99.8, 99.9],
            "close": [101.0, 100.2, 101.5, 100.0, 101.8],
            "volume": [100] * 5,
        }
    )
    bars, features = feature_inputs(frame)
    config, version = load_pattern_detector_config(STRATEGY_PATH)
    detector = H1H2Detector(config, strategy_version=version)

    events = []
    for bar, feature, context in zip(
        bars,
        features.itertuples(index=False),
        bull_contexts(features),
        strict=True,
    ):
        events.extend(detector.update(bar, feature._asdict(), context))

    assert [event.pattern_type for event in events] == [PatternType.H1, PatternType.H2]
    assert [event.signal_index for event in events] == [2, 4]
    assert events[0].start_index == 1
    assert events[1].trigger_price == 102.0
    assert events[1].metadata["attempt_number"] == 2
    assert events[1].metadata["quality_score_is_probability"] is False
    assert 0 <= events[1].confidence_score <= 1
    conditions = [transition.condition for transition in detector.debug_log]
    assert conditions == [
        "bear_bar_or_lower_low",
        "first_up_attempt",
        "new_down_leg_after_h1",
        "second_attempt_armed",
        "second_up_attempt",
    ]


def test_l1_l2_state_machine_is_a_bearish_mirror() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=5, freq="1min"),
            "open": [102.0, 101.0, 101.8, 100.5, 102.0],
            "high": [103.0, 102.0, 101.9, 102.2, 102.1],
            "low": [100.0, 100.5, 100.2, 100.3, 99.8],
            "close": [101.0, 101.8, 100.5, 102.0, 100.0],
            "volume": [100] * 5,
        }
    )
    bars, features = feature_inputs(frame)
    config, version = load_pattern_detector_config(STRATEGY_PATH)
    detector = L1L2Detector(config, strategy_version=version)

    events = []
    for bar, feature, context in zip(
        bars,
        features.itertuples(index=False),
        bear_contexts(features),
        strict=True,
    ):
        events.extend(detector.update(bar, feature._asdict(), context))

    assert [event.pattern_type for event in events] == [PatternType.L1, PatternType.L2]
    assert [event.signal_index for event in events] == [2, 4]
    assert events[1].trigger_price == 99.8
    assert [transition.condition for transition in detector.debug_log] == [
        "bull_bar_or_higher_high",
        "first_down_attempt",
        "new_up_leg_after_l1",
        "second_attempt_armed",
        "second_down_attempt",
    ]


def test_consecutive_up_attempts_do_not_create_h2_without_new_down_leg() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=5, freq="1min"),
            "open": [100.0, 101.0, 100.2, 101.0, 101.5],
            "high": [102.0, 101.5, 101.8, 102.0, 102.2],
            "low": [99.0, 100.0, 100.1, 100.5, 101.0],
            "close": [101.0, 100.2, 101.5, 101.8, 102.0],
            "volume": [100] * 5,
        }
    )
    bars, features = feature_inputs(frame)
    config, version = load_pattern_detector_config(STRATEGY_PATH)
    detector = H1H2Detector(config, strategy_version=version)

    events = []
    for bar, feature, context in zip(
        bars,
        features.itertuples(index=False),
        bull_contexts(features),
        strict=True,
    ):
        events.extend(detector.update(bar, feature._asdict(), context))

    assert [event.pattern_type for event in events] == [PatternType.H1]


def test_pattern_engine_is_causal_and_resets_on_context_loss() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=6, freq="1min"),
            "open": [100.0, 101.0, 100.2, 101.5, 100.0, 101.8],
            "high": [102.0, 101.5, 101.8, 101.6, 102.0, 103.0],
            "low": [99.0, 100.0, 100.1, 99.8, 99.9, 101.0],
            "close": [101.0, 100.2, 101.5, 100.0, 101.8, 102.8],
            "volume": [100] * 6,
        }
    )
    bars, features = feature_inputs(frame)
    contexts = bull_contexts(features)
    config, version = load_pattern_detector_config(STRATEGY_PATH)
    engine = FirstSecondEntryPatternEngine(config, strategy_version=version)

    full_events = engine.detect(bars, features, contexts)
    prefix_events = engine.detect(bars[:5], features.iloc[:5], contexts[:5])

    assert full_events[: len(prefix_events)] == prefix_events
    lost_context = contexts.copy()
    lost_context[3] = lost_context[3].model_copy(
        update={
            "regime": MarketRegime.TRADING_RANGE,
            "always_in": AlwaysInState.NEUTRAL,
            "trend_score": 0.0,
        }
    )
    reset_events = engine.detect(bars, features, lost_context)
    assert PatternType.H2 not in [event.pattern_type for event in reset_events]


def test_pattern_configuration_controls_expiry_and_debug_log() -> None:
    loaded, version = load_pattern_detector_config(STRATEGY_PATH)
    config = loaded.model_copy(
        update={
            "pullback_max_bars": 2,
            "debug_transitions": False,
        }
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=4, freq="1min"),
            "open": [100.0, 101.0, 100.8, 100.6],
            "high": [102.0, 101.5, 101.4, 101.3],
            "low": [99.0, 100.0, 99.9, 99.8],
            "close": [101.0, 100.8, 100.6, 100.4],
            "volume": [100] * 4,
        }
    )
    bars, features = feature_inputs(frame)
    detector = H1H2Detector(config, strategy_version=version)
    events = []
    for bar, feature, context in zip(
        bars,
        features.itertuples(index=False),
        bull_contexts(features),
        strict=True,
    ):
        events.extend(detector.update(bar, feature._asdict(), context))

    assert events == []
    assert detector.debug_log == []
    assert detector.start_index is None


def test_pattern_configuration_rejects_inverted_pullback_limits() -> None:
    loaded, _ = load_pattern_detector_config(STRATEGY_PATH)
    values = loaded.model_dump()
    values.update({"pullback_min_bars": 3, "pullback_max_bars": 2})

    try:
        PatternDetectorConfig.model_validate(values)
    except ValidationError as error:
        assert "pullback_max_bars cannot be below" in str(error)
    else:
        raise AssertionError("inverted pullback limits should fail validation")
