from pathlib import Path

import yaml

from brooks_trader.strategy import StrategyModuleSelection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_strategy_configuration_has_version_and_execution_policy() -> None:
    with (PROJECT_ROOT / "config" / "strategy.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    assert config["strategy"]["version"] == "0.7.0-phase7"
    assert config["execution"]["same_bar_stop_target_policy"] == "adverse"
    assert config["entry"]["second_entries_only"] is True
    assert config["backtest"] == {
        "initial_cash": 100000.0,
        "allow_multiple_positions": False,
        "pending_order_expiry_bars": 1,
        "close_open_position_at_end": True,
    }
    assert config["statistics"]["minimum_probability_sample"] == 30
    assert config["statistics"]["session_timezone"] == "America/New_York"


def test_every_configured_market_has_positive_tick_size() -> None:
    with (PROJECT_ROOT / "config" / "markets.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    assert config["markets"]
    assert all(market["tick_size"] > 0 for market in config["markets"].values())


def test_strategy_module_selection_defaults_preserve_current_pipeline() -> None:
    selection = StrategyModuleSelection.from_values(
        {"ema_alignment_filter": True},
        overrides={"ema_alignment_filter": False},
    )

    assert selection.ema_alignment_filter is False
    assert selection.h2_with_trend is True
    assert "h2_with_trend" in selection.enabled_ids()
