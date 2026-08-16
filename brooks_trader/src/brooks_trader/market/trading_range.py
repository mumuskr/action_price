"""Trading-range classification helpers."""

from brooks_trader.market.trend import MarketContextConfig


def is_trading_range_score(trend_score: float, config: MarketContextConfig) -> bool:
    """Return whether the score lies in the configured neutral range band."""
    return abs(trend_score) <= config.trading_range_threshold
