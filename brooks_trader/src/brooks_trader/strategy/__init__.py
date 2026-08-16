"""Public Phase 5 setup and strategy decision API."""

from brooks_trader.strategy.brooks_strategy import BrooksStrategy
from brooks_trader.strategy.catalog import (
    ModuleStatus,
    StrategyModule,
    StrategyModuleSelection,
    get_strategy_module,
    strategy_module_catalog,
)
from brooks_trader.strategy.setup_engine import (
    MarketExecutionConfig,
    SetupConfig,
    SetupEngine,
    SetupEngineConfig,
    load_setup_engine_config,
)
from brooks_trader.strategy.trader_equation import ExpectedValue, calculate_expected_value

__all__ = [
    "BrooksStrategy",
    "ExpectedValue",
    "MarketExecutionConfig",
    "ModuleStatus",
    "SetupConfig",
    "SetupEngine",
    "SetupEngineConfig",
    "StrategyModule",
    "StrategyModuleSelection",
    "calculate_expected_value",
    "get_strategy_module",
    "load_setup_engine_config",
    "strategy_module_catalog",
]
