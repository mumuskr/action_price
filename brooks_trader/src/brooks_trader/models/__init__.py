"""Stable domain contracts shared by the research system."""

from brooks_trader.models.bar import Bar, BarFeatures
from brooks_trader.models.common import Direction
from brooks_trader.models.knowledge import (
    BookParagraph,
    BrooksRule,
    BrooksRuleSource,
    KnowledgeChunk,
    RuleDirection,
    RuleStatus,
)
from brooks_trader.models.market_state import AlwaysInState, MarketRegime, MarketState
from brooks_trader.models.pattern import PatternEvent, PatternType
from brooks_trader.models.setup import SetupEvaluation, SetupType, TradeSetup
from brooks_trader.models.signal import SignalType, StrategySignal
from brooks_trader.models.swing import SwingPoint, SwingType
from brooks_trader.models.trade import (
    Order,
    OrderStatus,
    OrderType,
    TradeIntent,
    TradeRecord,
)

__all__ = [
    "AlwaysInState",
    "Bar",
    "BarFeatures",
    "BookParagraph",
    "BrooksRule",
    "BrooksRuleSource",
    "Direction",
    "KnowledgeChunk",
    "MarketRegime",
    "MarketState",
    "Order",
    "OrderStatus",
    "OrderType",
    "PatternEvent",
    "PatternType",
    "RuleDirection",
    "RuleStatus",
    "SetupEvaluation",
    "SetupType",
    "SignalType",
    "StrategySignal",
    "SwingPoint",
    "SwingType",
    "TradeIntent",
    "TradeRecord",
    "TradeSetup",
]
