from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from brooks_trader.backtest import (
    BacktestEngine,
    ClosedTrade,
    ExecutionEventType,
    PaperBroker,
    Portfolio,
)
from brooks_trader.models import (
    AlwaysInState,
    Bar,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = PROJECT_ROOT / "config" / "strategy.yaml"
MARKETS_PATH = PROJECT_ROOT / "config" / "markets.yaml"
VERSION = "0.7.0-phase7"


def make_signal(
    direction: Direction = Direction.LONG,
    *,
    signal_index: int = 3,
) -> StrategySignal:
    timestamp = datetime(2026, 1, 2, 14, 33, tzinfo=UTC)
    context = MarketState(
        timestamp=timestamp,
        bar_index=signal_index,
        regime=(
            MarketRegime.BULL_TREND if direction == Direction.LONG else MarketRegime.BEAR_TREND
        ),
        always_in=(
            AlwaysInState.ALWAYS_IN_LONG
            if direction == Direction.LONG
            else AlwaysInState.ALWAYS_IN_SHORT
        ),
        trend_score=0.6 if direction == Direction.LONG else -0.6,
        pressure_score=0.2 if direction == Direction.LONG else -0.2,
        strategy_version=VERSION,
    )
    pattern = PatternEvent(
        pattern_type=PatternType.H2 if direction == Direction.LONG else PatternType.L2,
        direction=direction,
        start_index=1,
        signal_index=signal_index,
        signal_time=timestamp,
        trigger_price=101.0 if direction == Direction.LONG else 99.0,
        context=context,
        confidence_score=0.7,
        metadata={"attempt_number": 2},
        strategy_version=VERSION,
    )
    setup = TradeSetup(
        setup_type=(
            SetupType.H2_WITH_TREND if direction == Direction.LONG else SetupType.L2_WITH_TREND
        ),
        direction=direction,
        evaluated_at=timestamp,
        signal_bar_index=signal_index,
        pattern=pattern,
        market_state=context,
        pattern_score=0.7,
        context_score=0.6,
        entry=101.0 if direction == Direction.LONG else 99.0,
        initial_stop=99.0 if direction == Direction.LONG else 101.0,
        target=105.0 if direction == Direction.LONG else 95.0,
        invalidation="initial stop",
        reasons=["second_entry_pattern", "market_regime_with_trend"],
        metadata={
            "signal_body_ratio": 0.7,
            "signal_close_location": 0.8 if direction == Direction.LONG else 0.2,
            "ema_slope_ratio": 0.0003 if direction == Direction.LONG else -0.0003,
            "volatility_range_ratio": 0.005,
        },
        strategy_version=VERSION,
    )
    return StrategySignal(
        signal_type=SignalType.SECOND_ENTRY_WITH_TREND,
        created_at=timestamp,
        signal_bar_index=signal_index,
        direction=direction,
        setup=setup,
        reasons=list(setup.reasons),
        strategy_version=VERSION,
    )


def make_bar(index: int, open_: float, high: float, low: float, close: float) -> Bar:
    return Bar(
        timestamp=datetime(2026, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
    )


def test_signal_bar_cannot_fill_and_entry_starts_on_next_bar() -> None:
    broker = PaperBroker(tick_size=0.01, pending_order_expiry_bars=2)
    broker.submit_signal(make_signal(), quantity=10, submitted_index=3)

    assert broker.process_bar(make_bar(3, 100, 102, 99.5, 101), bar_index=3) == []
    events = broker.process_bar(make_bar(4, 100.5, 101.2, 100, 101.1), bar_index=4)

    assert [event.event_type for event in events] == [ExecutionEventType.ENTRY_FILLED]
    assert events[0].fill_price == 101.0


def test_gap_entry_and_slippage_are_adverse_for_both_directions() -> None:
    long_broker = PaperBroker(tick_size=0.01, slippage_ticks=1)
    long_broker.submit_signal(make_signal(), quantity=1, submitted_index=3)
    long_event = long_broker.process_bar(make_bar(4, 102.0, 102.2, 101.5, 102.0), bar_index=4)[0]

    short_broker = PaperBroker(tick_size=0.01, slippage_ticks=1)
    short_broker.submit_signal(make_signal(Direction.SHORT), quantity=1, submitted_index=3)
    short_event = short_broker.process_bar(make_bar(4, 98.0, 98.5, 97.8, 98.0), bar_index=4)[0]

    assert long_event.fill_price == 102.01
    assert short_event.fill_price == 97.99


def test_short_entry_bar_can_close_at_stop_with_adverse_policy() -> None:
    broker = PaperBroker(
        tick_size=0.01,
        same_bar_stop_target_policy="adverse",
        pending_order_expiry_bars=2,
    )
    broker.submit_signal(make_signal(Direction.SHORT), quantity=1, submitted_index=3)

    events = broker.process_bar(make_bar(4, 102, 103, 94, 100), bar_index=4)

    assert [event.event_type for event in events] == [
        ExecutionEventType.ENTRY_FILLED,
        ExecutionEventType.TRADE_CLOSED,
    ]
    assert events[1].fill_price == 101
    assert events[1].trade is not None
    assert events[1].trade.exit_reason == "STOP_LOSS"


def test_same_bar_stop_target_conflict_uses_adverse_policy() -> None:
    broker = PaperBroker(
        tick_size=0.01,
        same_bar_stop_target_policy="adverse",
        pending_order_expiry_bars=2,
    )
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)

    events = broker.process_bar(make_bar(4, 101, 106, 98, 103), bar_index=4)

    assert [event.event_type for event in events] == [
        ExecutionEventType.ENTRY_FILLED,
        ExecutionEventType.TRADE_CLOSED,
    ]
    assert events[1].fill_price == 99.0
    assert events[1].trade is not None
    assert events[1].trade.exit_reason == "STOP_LOSS"


def test_pre_entry_open_does_not_create_a_stop_gap_fill() -> None:
    broker = PaperBroker(
        tick_size=0.01,
        slippage_ticks=1,
        same_bar_stop_target_policy="adverse",
        pending_order_expiry_bars=2,
    )
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)

    events = broker.process_bar(make_bar(4, 98, 102, 97, 100), bar_index=4)

    assert events[0].event_type == ExecutionEventType.ENTRY_FILLED
    assert events[1].event_type == ExecutionEventType.TRADE_CLOSED
    assert events[1].fill_price == 98.99


def test_gap_entry_beyond_target_exits_immediately_at_open() -> None:
    broker = PaperBroker(
        tick_size=0.01,
        slippage_ticks=1,
        same_bar_stop_target_policy="adverse",
        pending_order_expiry_bars=2,
    )
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)

    events = broker.process_bar(make_bar(4, 106, 107, 98, 100), bar_index=4)

    assert events[0].fill_price == 106.01
    assert events[1].fill_price == 105.99
    assert events[1].trade is not None
    assert events[1].trade.exit_reason == "TARGET"


def test_short_stop_gap_is_filled_at_worse_open_with_slippage() -> None:
    broker = PaperBroker(tick_size=0.01, slippage_ticks=1, pending_order_expiry_bars=2)
    broker.submit_signal(make_signal(Direction.SHORT), quantity=1, submitted_index=3)
    broker.process_bar(make_bar(4, 99, 100, 98, 98.5), bar_index=4)

    events = broker.process_bar(make_bar(5, 102, 103, 101.5, 102.5), bar_index=5)

    assert events[0].event_type == ExecutionEventType.TRADE_CLOSED
    assert events[0].fill_price == 102.01


def test_open_gap_to_target_is_resolved_before_intrabar_stop() -> None:
    broker = PaperBroker(
        tick_size=0.01,
        same_bar_stop_target_policy="adverse",
        pending_order_expiry_bars=2,
    )
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)
    broker.process_bar(make_bar(4, 101, 102, 100, 101), bar_index=4)

    events = broker.process_bar(make_bar(5, 106, 107, 98, 100), bar_index=5)

    assert events[0].trade is not None
    assert events[0].trade.exit_reason == "TARGET"
    assert events[0].fill_price == 106


def test_pending_entry_expires_after_configured_eligible_bar() -> None:
    broker = PaperBroker(tick_size=0.01, pending_order_expiry_bars=1)
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)

    events = broker.process_bar(make_bar(4, 100, 100.5, 99, 100), bar_index=4)

    assert events[0].event_type == ExecutionEventType.ORDER_CANCELLED
    assert events[0].order.rejection_reason == "PENDING_ORDER_EXPIRED"
    assert broker.pending_entry is None


def test_portfolio_records_commission_mfe_mae_and_r_values() -> None:
    signal = make_signal()
    closed = ClosedTrade(
        trade_id="trade-1",
        signal=signal,
        quantity=10,
        entry_index=4,
        entry_time=make_bar(4, 101, 102, 100, 101).timestamp,
        entry_price=101,
        stop_price=99,
        target_price=105,
        exit_index=6,
        exit_time=make_bar(6, 104, 105, 103, 105).timestamp,
        exit_price=105,
        exit_reason="TARGET",
        mfe=4.5,
        mae=1.0,
    )
    portfolio = Portfolio(
        initial_cash=100_000,
        risk_per_trade=0.005,
        point_value=1,
        commission_per_trade=2,
        slippage_ticks=1,
    )

    pnl = portfolio.realize(closed)
    record = portfolio.build_trade_record(closed, symbol="SPY", timeframe="1m", pnl=pnl)

    assert pnl == 38
    assert record.initial_risk == 20
    assert record.quantity == 10
    assert record.point_value == 1
    assert record.slippage_ticks == 1
    assert record.gross_pnl == 40
    assert record.pnl_r == 1.9
    assert record.mfe_r == 2.25
    assert record.mae_r == 0.5
    assert record.commission == 2
    assert record.bars_held == 3
    assert record.strategy_version == VERSION


def test_force_close_uses_last_close_and_cancels_unfilled_order() -> None:
    broker = PaperBroker(tick_size=0.01, slippage_ticks=1, pending_order_expiry_bars=3)
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)
    final = make_bar(4, 100, 100.5, 99.5, 100.2)

    events = broker.force_close(final, bar_index=4)

    assert events[0].event_type == ExecutionEventType.ORDER_CANCELLED
    assert events[0].order.rejection_reason == "END_OF_DATA"


def test_force_close_open_position_uses_adverse_market_slippage() -> None:
    broker = PaperBroker(tick_size=0.01, slippage_ticks=1, pending_order_expiry_bars=2)
    broker.submit_signal(make_signal(), quantity=1, submitted_index=3)
    broker.process_bar(make_bar(4, 101, 102, 100, 101.5), bar_index=4)
    final = make_bar(5, 102, 103, 101.5, 102.5)

    events = broker.force_close(final, bar_index=5)

    assert events[0].event_type == ExecutionEventType.TRADE_CLOSED
    assert events[0].fill_price == 102.49
    assert events[0].order.order_type.value == "MARKET"
    assert events[0].trade is not None
    assert events[0].trade.exit_reason == "END_OF_DATA"


def test_changing_future_bars_does_not_change_past_pipeline_artifacts() -> None:
    frame = pd.read_parquet(
        PROJECT_ROOT / "data/processed/symbol=SPY/timeframe=5m/bars.parquet"
    ).iloc[:300]
    split = 200
    engine = BacktestEngine.from_config(
        symbol="SPY",
        timeframe="5m",
        strategy_path=STRATEGY_PATH,
        markets_path=MARKETS_PATH,
    )

    prefix = engine.run(frame.iloc[:split])
    changed = frame.copy()
    future = changed.index >= split
    changed.loc[future, "open"] *= 2
    changed.loc[future, "high"] *= 2
    changed.loc[future, "low"] *= 2
    changed.loc[future, "close"] *= 2
    full = engine.run(changed)

    assert full.market_states[:split] == prefix.market_states
    assert (
        tuple(pattern for pattern in full.patterns if pattern.signal_index < split)
        == prefix.patterns
    )
    assert tuple(signal for signal in full.signals if signal.signal_bar_index < split) == (
        prefix.signals
    )
