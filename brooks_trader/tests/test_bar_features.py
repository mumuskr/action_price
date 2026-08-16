from pathlib import Path

import pandas as pd
import pytest

from brooks_trader.features import (
    BarFeatureConfig,
    calculate_bar_features,
    calculate_body_ratio,
    calculate_ema20,
    calculate_ema_slope,
    calculate_overlap,
    calculate_tail_ratio,
    is_bear_bar,
    is_bull_bar,
    is_doji,
    is_inside_bar,
    is_outside_bar,
    is_trend_bar,
    load_bar_feature_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config() -> BarFeatureConfig:
    return load_bar_feature_config(PROJECT_ROOT / "config" / "strategy.yaml")


def test_scalar_bar_classifications_and_ratios() -> None:
    assert calculate_body_ratio(100.0, 103.0, 4.0) == pytest.approx(0.75)
    assert calculate_body_ratio(100.0, 100.0, 0.0) == 0.0
    assert calculate_tail_ratio(1.0, 4.0) == pytest.approx(0.25)
    assert calculate_tail_ratio(0.0, 0.0) == 0.0
    assert is_bull_bar(100.0, 101.0)
    assert is_bear_bar(101.0, 100.0)
    assert is_doji(0.2, threshold=0.2)
    assert is_trend_bar(
        body_ratio=0.8,
        close_location=0.9,
        bull_bar=True,
        bear_bar=False,
        strong_body_ratio=0.65,
        strong_close_threshold=0.75,
    )


def test_inside_outside_and_overlap_are_explicit() -> None:
    assert is_inside_bar(104.0, 97.0, 105.0, 95.0)
    assert not is_inside_bar(105.0, 95.0, 105.0, 95.0)
    assert is_outside_bar(106.0, 94.0, 105.0, 95.0)
    assert not is_outside_bar(105.0, 95.0, 105.0, 95.0)
    assert calculate_overlap(104.0, 97.0, 105.0, 95.0) == pytest.approx(0.7)
    assert calculate_overlap(110.0, 106.0, 105.0, 100.0) == 0.0


def test_ema_and_slope_use_only_current_and_past_values() -> None:
    closes = pd.Series([1.0, 2.0, 3.0])

    ema = calculate_ema20(closes, period=2)
    slope = calculate_ema_slope(ema, lookback=1)

    assert ema.tolist() == pytest.approx([1.0, 5 / 3, 23 / 9])
    assert pd.isna(slope.iloc[0])
    assert slope.iloc[1] == pytest.approx(2 / 3)


def test_calculate_bar_features_covers_structure_and_zero_range(
    config: BarFeatureConfig,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=4, freq="1min"),
            "open": [97.0, 99.0, 98.0, 106.0],
            "high": [105.0, 104.0, 106.0, 106.0],
            "low": [95.0, 97.0, 94.0, 106.0],
            "close": [104.0, 99.5, 105.0, 106.0],
            "volume": [100, 100, 100, 0],
        }
    )

    result = calculate_bar_features(frame, config=config)

    assert result.loc[0, "bull_bar"]
    assert result.loc[0, "trend_bar"]
    assert result.loc[1, "inside_bar"]
    assert result.loc[1, "lower_high"]
    assert result.loc[1, "higher_low"]
    assert result.loc[2, "outside_bar"]
    assert result.loc[2, "higher_high"]
    assert result.loc[2, "lower_low"]
    assert result.loc[3, "range"] == 0.0
    assert result.loc[3, "close_location"] == 0.5
    assert result.loc[3, "doji"]
    assert not result.loc[3, "trend_bar"]


def test_changing_a_future_bar_does_not_change_prior_features(
    config: BarFeatureConfig,
) -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=8, freq="1min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100] * 8,
        }
    )
    changed = base.copy()
    changed.loc[7, ["open", "high", "low", "close"]] = [200.0, 220.0, 190.0, 210.0]

    original_features = calculate_bar_features(base, config=config)
    changed_features = calculate_bar_features(changed, config=config)

    pd.testing.assert_frame_equal(
        original_features.iloc[:7].reset_index(drop=True),
        changed_features.iloc[:7].reset_index(drop=True),
    )
