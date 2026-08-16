"""Single-position portfolio accounting for Phase 6 backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

from brooks_trader.backtest.broker import ClosedTrade
from brooks_trader.models import Direction, TradeRecord


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Portfolio balance after one completed trade."""

    trade_id: str
    cash: float
    realized_pnl: float


class Portfolio:
    """Size positions from current equity and realize closed-trade PnL."""

    def __init__(
        self,
        *,
        initial_cash: float,
        risk_per_trade: float,
        point_value: float,
        commission_per_trade: float = 0.0,
        slippage_ticks: int = 0,
    ) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not isfinite(risk_per_trade) or not 0 < risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be between zero and one")
        if not isfinite(point_value) or point_value <= 0:
            raise ValueError("point_value must be finite and positive")
        if not isfinite(commission_per_trade) or commission_per_trade < 0:
            raise ValueError("commission_per_trade must be finite and non-negative")
        if slippage_ticks < 0:
            raise ValueError("slippage_ticks cannot be negative")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.commission_per_trade = commission_per_trade
        self.slippage_ticks = slippage_ticks
        self.snapshots: list[PortfolioSnapshot] = []

    def size_for_entry(self, *, entry_price: float, stop_price: float) -> float:
        """Return whole units whose configured initial risk does not exceed equity risk."""
        risk_per_unit = abs(entry_price - stop_price) * self.point_value
        if risk_per_unit <= 0 or not isfinite(risk_per_unit):
            raise ValueError("entry and stop must define a finite positive risk")
        quantity = floor(self.cash * self.risk_per_trade / risk_per_unit)
        return float(max(0, quantity))

    def realize(self, trade: ClosedTrade) -> float:
        """Apply directional price PnL and one round-trip commission to cash."""
        direction = 1.0 if trade.signal.direction == Direction.LONG else -1.0
        gross = (
            direction * (trade.exit_price - trade.entry_price) * trade.quantity * self.point_value
        )
        pnl = gross - self.commission_per_trade
        self.cash += pnl
        self.snapshots.append(
            PortfolioSnapshot(trade_id=trade.trade_id, cash=self.cash, realized_pnl=pnl)
        )
        return pnl

    def build_trade_record(
        self,
        trade: ClosedTrade,
        *,
        symbol: str,
        timeframe: str,
        pnl: float,
    ) -> TradeRecord:
        """Convert one closed execution into the immutable audit contract."""
        setup = trade.signal.setup
        initial_risk_points = abs(trade.entry_price - trade.stop_price)
        initial_risk_cash = initial_risk_points * trade.quantity * self.point_value
        pnl_r = pnl / initial_risk_cash if initial_risk_cash > 0 else 0.0
        return TradeRecord(
            trade_id=trade.trade_id,
            symbol=symbol,
            timeframe=timeframe,
            setup=setup.setup_type.value,
            pattern_type=setup.pattern.pattern_type,
            direction=trade.signal.direction,
            market_regime=setup.market_state.regime,
            pattern_score=setup.pattern_score,
            context_score=setup.context_score,
            signal_body_ratio=float(setup.metadata["signal_body_ratio"]),
            signal_close_location=float(setup.metadata["signal_close_location"]),
            ema_slope_ratio=float(setup.metadata["ema_slope_ratio"]),
            volatility_range_ratio=float(setup.metadata["volatility_range_ratio"]),
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            point_value=self.point_value,
            slippage_ticks=self.slippage_ticks,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            exit_time=trade.exit_time,
            exit_price=trade.exit_price,
            initial_risk=initial_risk_cash,
            gross_pnl=pnl + self.commission_per_trade,
            pnl=pnl,
            commission=self.commission_per_trade,
            pnl_r=pnl_r,
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_r=trade.mfe / initial_risk_points,
            mae_r=trade.mae / initial_risk_points,
            bars_held=trade.exit_index - trade.entry_index + 1,
            signal_bar_index=trade.signal.signal_bar_index,
            market_state=setup.market_state,
            pattern_metadata={
                **setup.pattern.metadata,
                "pattern_type": setup.pattern.pattern_type.value,
                "pattern_start_index": setup.pattern.start_index,
                "pattern_signal_index": setup.pattern.signal_index,
            },
            entry_reason="; ".join(trade.signal.reasons),
            exit_reason=trade.exit_reason,
            strategy_version=trade.signal.strategy_version,
        )
