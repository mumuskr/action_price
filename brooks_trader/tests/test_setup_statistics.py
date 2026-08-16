from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from brooks_trader.backtest import ClosedTrade, Portfolio
from brooks_trader.models import (
    AlwaysInState,
    Direction,
    MarketRegime,
    MarketState,
    PatternEvent,
    PatternType,
    SetupType,
    SignalType,
    StrategySignal,
    TradeSetup,
)
from brooks_trader.statistics import (
    EmaSlopeBucket,
    SessionBucket,
    SignalQualityBucket,
    StatisticsScope,
    VolatilityRegime,
    calculate_setup_statistics,
    classify_trade_conditions,
    find_empirical_probability,
    load_setup_statistics_config,
    statistics_to_frame,
    write_setup_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.7.0-phase7"


def make_trade(
    index: int,
    *,
    direction: Direction,
    regime: MarketRegime,
    pnl_points: float,
    body_ratio: float = 0.7,
    close_location: float | None = None,
    ema_slope_ratio: float | None = None,
    volatility_range_ratio: float = 0.005,
    hour_utc: int = 15,
):
    timestamp = datetime(2026, 1, 5, hour_utc, 0, tzinfo=UTC) + timedelta(days=index)
    pattern_type = PatternType.H2 if direction == Direction.LONG else PatternType.L2
    setup_type = SetupType.H2_WITH_TREND if direction == Direction.LONG else SetupType.L2_WITH_TREND
    directional_score = 0.6 if direction == Direction.LONG else -0.6
    close_location = (
        close_location
        if close_location is not None
        else (0.8 if direction == Direction.LONG else 0.2)
    )
    ema_slope_ratio = (
        ema_slope_ratio
        if ema_slope_ratio is not None
        else (0.0003 if direction == Direction.LONG else -0.0003)
    )
    context = MarketState(
        timestamp=timestamp,
        bar_index=index,
        regime=regime,
        always_in=(
            AlwaysInState.ALWAYS_IN_LONG
            if direction == Direction.LONG
            else AlwaysInState.ALWAYS_IN_SHORT
        ),
        trend_score=directional_score,
        pressure_score=directional_score / 2,
        strategy_version=VERSION,
    )
    pattern = PatternEvent(
        pattern_type=pattern_type,
        direction=direction,
        start_index=max(0, index - 2),
        signal_index=index,
        signal_time=timestamp,
        trigger_price=101 if direction == Direction.LONG else 99,
        context=context,
        confidence_score=0.7,
        metadata={"attempt_number": 2},
        strategy_version=VERSION,
    )
    entry = 101.0 if direction == Direction.LONG else 99.0
    stop = 99.0 if direction == Direction.LONG else 101.0
    target = 105.0 if direction == Direction.LONG else 95.0
    setup = TradeSetup(
        setup_type=setup_type,
        direction=direction,
        evaluated_at=timestamp,
        signal_bar_index=index,
        pattern=pattern,
        market_state=context,
        pattern_score=0.7,
        context_score=0.6,
        entry=entry,
        initial_stop=stop,
        target=target,
        invalidation="initial stop",
        reasons=["second_entry_pattern"],
        metadata={
            "signal_body_ratio": body_ratio,
            "signal_close_location": close_location,
            "ema_slope_ratio": ema_slope_ratio,
            "volatility_range_ratio": volatility_range_ratio,
        },
        strategy_version=VERSION,
    )
    signal = StrategySignal(
        signal_type=SignalType.SECOND_ENTRY_WITH_TREND,
        created_at=timestamp,
        signal_bar_index=index,
        direction=direction,
        setup=setup,
        reasons=list(setup.reasons),
        strategy_version=VERSION,
    )
    exit_price = entry + pnl_points if direction == Direction.LONG else entry - pnl_points
    closed = ClosedTrade(
        trade_id=f"trade-{index}",
        signal=signal,
        quantity=1,
        entry_index=index + 1,
        entry_time=timestamp + timedelta(minutes=5),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        exit_index=index + 2,
        exit_time=timestamp + timedelta(minutes=10),
        exit_price=exit_price,
        exit_reason="TARGET" if pnl_points > 0 else "STOP_LOSS",
        mfe=max(0.0, pnl_points) + 0.5,
        mae=max(0.0, -pnl_points) + 0.25,
    )
    portfolio = Portfolio(initial_cash=100_000, risk_per_trade=0.005, point_value=1)
    pnl = portfolio.realize(closed)
    return portfolio.build_trade_record(closed, symbol="SPY", timeframe="5m", pnl=pnl)


def test_statistics_aggregate_exact_counts_r_metrics_and_drawdown() -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml").model_copy(
        update={"minimum_probability_sample": 3}
    )
    trades = [
        make_trade(
            0,
            direction=Direction.LONG,
            regime=MarketRegime.BULL_TREND,
            pnl_points=4,
        ),
        make_trade(
            1,
            direction=Direction.LONG,
            regime=MarketRegime.BULL_TREND,
            pnl_points=-2,
        ),
        make_trade(
            2,
            direction=Direction.SHORT,
            regime=MarketRegime.BEAR_TREND,
            pnl_points=-2,
        ),
    ]

    statistics = calculate_setup_statistics(trades, config=config)
    overall = next(item for item in statistics if item.scope == StatisticsScope.OVERALL)

    assert overall.total == 3
    assert overall.wins == 1
    assert overall.losses == 2
    assert overall.win_rate == pytest.approx(1 / 3)
    assert overall.probability_win == pytest.approx(1 / 3)
    assert overall.avg_win_r == 2
    assert overall.avg_loss_r == -1
    assert overall.expectancy_r == 0
    assert overall.profit_factor == 1
    assert overall.max_drawdown == 4
    assert overall.median_mfe_r == 0.25
    assert overall.median_mae_r == 1.125
    assert {item.scope for item in statistics} == set(StatisticsScope)


def test_probability_remains_unknown_below_configured_sample_minimum() -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml")
    trades = [
        make_trade(
            0,
            direction=Direction.LONG,
            regime=MarketRegime.BULL_TREND,
            pnl_points=4,
        )
    ]

    statistics = calculate_setup_statistics(trades, config=config)
    overall = next(item for item in statistics if item.scope == StatisticsScope.OVERALL)

    assert overall.win_rate == 1
    assert overall.probability_win is None
    assert overall.profit_factor is None
    assert (
        find_empirical_probability(
            statistics,
            symbol="SPY",
            timeframe="5m",
            strategy_version=VERSION,
            pattern_type=PatternType.H2,
            market_regime=MarketRegime.BULL_TREND,
        )
        is None
    )


def test_condition_buckets_are_direction_aware_and_configurable() -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml")
    trade = make_trade(
        0,
        direction=Direction.SHORT,
        regime=MarketRegime.BEAR_TREND,
        pnl_points=4,
        body_ratio=0.9,
        close_location=0.1,
        ema_slope_ratio=-0.001,
        volatility_range_ratio=0.01,
        hour_utc=20,
    )

    conditions = classify_trade_conditions(trade, config=config)

    assert conditions.signal_bar_quality == SignalQualityBucket.STRONG
    assert conditions.ema_slope_bucket == EmaSlopeBucket.STRONG
    assert conditions.volatility_regime == VolatilityRegime.HIGH
    assert conditions.session == SessionBucket.CLOSE


def test_statistics_parquet_round_trip_and_empty_schema(tmp_path: Path) -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml")
    trade = make_trade(
        0,
        direction=Direction.LONG,
        regime=MarketRegime.STRONG_BULL_TREND,
        pnl_points=4,
    )
    statistics = calculate_setup_statistics([trade], config=config)
    destination = write_setup_statistics(statistics, tmp_path / "setup_statistics.parquet")
    restored = pd.read_parquet(destination)

    assert not restored.empty
    assert set(restored["strategy_version"]) == {VERSION}
    assert restored["probability_win"].isna().all()
    assert "median_mfe_r" in statistics_to_frame([]).columns


def test_empty_trade_input_creates_no_fabricated_statistics() -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml")

    assert calculate_setup_statistics([], config=config) == []


def test_open_trade_cannot_be_treated_as_a_breakeven() -> None:
    config = load_setup_statistics_config(PROJECT_ROOT / "config/strategy.yaml")
    trade = make_trade(
        0,
        direction=Direction.LONG,
        regime=MarketRegime.BULL_TREND,
        pnl_points=4,
    ).model_copy(
        update={
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl": None,
            "pnl_r": None,
        }
    )

    with pytest.raises(ValueError, match="completed trades with PnL"):
        calculate_setup_statistics([trade], config=config)
