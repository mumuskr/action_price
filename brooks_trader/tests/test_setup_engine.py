from pathlib import Path

import pandas as pd

from brooks_trader.data.loader import bars_from_frame
from brooks_trader.features import calculate_bar_features, load_bar_feature_config
from brooks_trader.models import (
    AlwaysInState,
    Direction,
    MarketRegime,
    MarketState,
    PatternEvent,
    PatternType,
)
from brooks_trader.strategy import (
    BrooksStrategy,
    SetupEngine,
    calculate_expected_value,
    load_setup_engine_config,
)
from brooks_trader.strategy.setup_engine import ExitConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = PROJECT_ROOT / "config" / "strategy.yaml"
MARKETS_PATH = PROJECT_ROOT / "config" / "markets.yaml"
VERSION = "0.7.0-phase7"


def make_history(direction: Direction):
    if direction == Direction.LONG:
        opens = [96.0, 97.0, 98.0, 99.0, 100.0, 101.0]
        highs = [98.0, 99.0, 100.0, 101.0, 102.0, 103.0]
        lows = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
        closes = [97.5, 98.5, 99.5, 100.5, 101.5, 102.8]
        regime = MarketRegime.BULL_TREND
        always_in = AlwaysInState.ALWAYS_IN_LONG
        score = 0.6
        pattern_type = PatternType.H2
    else:
        opens = [104.0, 103.0, 102.0, 101.0, 100.0, 99.0]
        highs = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        lows = [102.0, 101.0, 100.0, 99.0, 98.0, 97.0]
        closes = [102.5, 101.5, 100.5, 99.5, 98.5, 97.2]
        regime = MarketRegime.BEAR_TREND
        always_in = AlwaysInState.ALWAYS_IN_SHORT
        score = -0.6
        pattern_type = PatternType.L2
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=6, freq="1min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * 6,
        }
    )
    bars = bars_from_frame(frame)
    features = calculate_bar_features(frame, config=load_bar_feature_config(STRATEGY_PATH))
    contexts = [
        MarketState(
            timestamp=row.timestamp,
            bar_index=int(row.bar_index),
            regime=regime,
            always_in=always_in,
            trend_score=score,
            ema_score=score,
            structure_score=score,
            pressure_score=score / 2,
            overlap_score=score / 2,
            breakout_score=score / 2,
            strategy_version=VERSION,
        )
        for row in features.itertuples(index=False)
    ]
    pattern = PatternEvent(
        pattern_type=pattern_type,
        direction=direction,
        start_index=3,
        signal_index=5,
        signal_time=bars[5].timestamp,
        trigger_price=bars[5].high if direction == Direction.LONG else bars[5].low,
        context=contexts[5],
        confidence_score=0.7,
        metadata={"attempt_number": 2, "quality_score_is_probability": False},
        strategy_version=VERSION,
    )
    return bars, features, contexts, pattern


def make_engine() -> SetupEngine:
    config, version = load_setup_engine_config(STRATEGY_PATH, MARKETS_PATH, symbol="SPY")
    return SetupEngine(config, strategy_version=version)


def test_accepts_long_h2_and_strategy_emits_signal_without_order() -> None:
    bars, features, contexts, pattern = make_history(Direction.LONG)

    evaluation = make_engine().evaluate(pattern, bars, features, contexts)
    signal = BrooksStrategy(strategy_version=VERSION).evaluate(evaluation)

    assert evaluation.accepted
    assert evaluation.setup is not None
    setup = evaluation.setup
    assert setup.entry == 103.01
    assert setup.initial_stop == 97.99
    assert setup.target == 113.05
    assert setup.metadata["probability_win"] is None
    assert setup.metadata["expected_value"] is None
    assert signal is not None
    assert signal.setup == setup
    assert not hasattr(signal, "quantity")
    assert not hasattr(signal, "order_type")


def test_accepts_short_l2_with_mirrored_prices() -> None:
    bars, features, contexts, pattern = make_history(Direction.SHORT)

    evaluation = make_engine().evaluate(pattern, bars, features, contexts)

    assert evaluation.accepted
    assert evaluation.setup is not None
    assert evaluation.setup.entry == 96.99
    assert evaluation.setup.initial_stop == 102.01
    assert evaluation.setup.target == 86.95


def test_first_entry_pattern_and_wrong_context_are_rejected() -> None:
    bars, features, contexts, pattern = make_history(Direction.LONG)
    first_entry = pattern.model_copy(update={"pattern_type": PatternType.H1})
    contexts[-1] = contexts[-1].model_copy(
        update={
            "regime": MarketRegime.TRADING_RANGE,
            "always_in": AlwaysInState.NEUTRAL,
            "trend_score": 0.0,
        }
    )
    first_entry = first_entry.model_copy(update={"context": contexts[-1]})

    evaluation = make_engine().evaluate(first_entry, bars, features, contexts)
    signal = BrooksStrategy(strategy_version=VERSION).evaluate(evaluation)

    assert not evaluation.accepted
    assert evaluation.setup is None
    assert "second_entry_pattern_required" in evaluation.rejection_reasons
    assert "market_regime_not_with_trend" in evaluation.rejection_reasons
    assert signal is None


def test_future_rows_do_not_change_existing_setup_evaluation() -> None:
    bars, features, contexts, pattern = make_history(Direction.LONG)
    engine = make_engine()
    original = engine.evaluate(pattern, bars, features, contexts)

    future_frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-02T14:36:00Z")],
            "open": [50.0],
            "high": [60.0],
            "low": [40.0],
            "close": [45.0],
            "volume": [100],
        }
    )
    future_bar = bars_from_frame(future_frame)[0]
    extended_bars = [*bars, future_bar]
    extended_features = pd.concat(
        [
            features,
            calculate_bar_features(
                future_frame, config=load_bar_feature_config(STRATEGY_PATH)
            ).assign(bar_index=6),
        ],
        ignore_index=True,
    )
    extended_contexts = [
        *contexts,
        contexts[-1].model_copy(update={"timestamp": future_bar.timestamp, "bar_index": 6}),
    ]

    changed = engine.evaluate(pattern, extended_bars, extended_features, extended_contexts)

    assert changed == original


def test_traders_equation_does_not_invent_probability() -> None:
    unknown = calculate_expected_value(probability_win=None, reward=2.0, risk=1.0)
    known = calculate_expected_value(probability_win=0.5, reward=2.0, risk=1.0, trading_cost=0.1)

    assert unknown.probability_win is None
    assert unknown.expected_value is None
    assert unknown.expected_value_r is None
    assert known.expected_value == 0.4
    assert known.expected_value_r == 0.4


def test_tick_size_comes_from_market_configuration() -> None:
    config, version = load_setup_engine_config(STRATEGY_PATH, MARKETS_PATH, symbol="ES")
    engine = SetupEngine(config, strategy_version=version)
    bars, features, contexts, pattern = make_history(Direction.LONG)

    evaluation = engine.evaluate(pattern, bars, features, contexts)

    assert evaluation.accepted
    assert evaluation.setup is not None
    assert evaluation.setup.entry == 103.25
    assert evaluation.setup.initial_stop == 97.75
    assert evaluation.setup.target == 114.25
    assert evaluation.setup.metadata["tick_size"] == 0.25


def test_quality_ema_and_pressure_failures_remain_auditable() -> None:
    bars, features, contexts, pattern = make_history(Direction.LONG)
    weak_pattern = pattern.model_copy(update={"confidence_score": 0.1})
    features.loc[5, "ema_slope"] = -1.0
    contexts[-1] = contexts[-1].model_copy(update={"pressure_score": -0.2})
    weak_pattern = weak_pattern.model_copy(update={"context": contexts[-1]})

    evaluation = make_engine().evaluate(weak_pattern, bars, features, contexts)

    assert not evaluation.accepted
    assert {
        "pattern_quality_below_minimum",
        "ema_not_aligned",
        "opposite_or_insufficient_pressure",
    } <= set(evaluation.rejection_reasons)
    assert evaluation.metadata["probability_win"] is None


def test_disabling_context_filter_keeps_direction_score_valid() -> None:
    bars, features, contexts, pattern = make_history(Direction.LONG)
    config, version = load_setup_engine_config(
        STRATEGY_PATH,
        MARKETS_PATH,
        symbol="SPY",
        module_overrides={"context_quality_filter": False, "pressure_filter": False},
    )
    contexts[-1] = contexts[-1].model_copy(update={"trend_score": -0.02, "pressure_score": 0.5})
    pattern = pattern.model_copy(update={"context": contexts[-1]})

    evaluation = SetupEngine(config, strategy_version=version).evaluate(
        pattern, bars, features, contexts
    )

    assert evaluation.accepted
    assert evaluation.setup is not None
    assert evaluation.setup.context_score == 0
    assert evaluation.setup.metadata["raw_context_score"] == -0.02


def test_exit_mode_and_reward_multiple_cannot_disagree() -> None:
    try:
        ExitConfig(mode="2R", reward_multiple=1.0)
    except ValueError as error:
        assert "reward_multiple must match" in str(error)
    else:
        raise AssertionError("inconsistent exit configuration should fail validation")
