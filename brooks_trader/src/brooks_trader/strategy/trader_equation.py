"""Statistics-backed Trader's Equation without invented probabilities."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExpectedValue(BaseModel):
    """Expected value result; unknown probability produces unknown EV."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probability_win: float | None = Field(default=None, ge=0, le=1)
    reward: float = Field(gt=0)
    risk: float = Field(gt=0)
    trading_cost: float = Field(default=0, ge=0)
    expected_value: float | None = None
    expected_value_r: float | None = None

    @model_validator(mode="after")
    def validate_calculated_values(self) -> "ExpectedValue":
        if self.probability_win is None:
            if self.expected_value is not None or self.expected_value_r is not None:
                raise ValueError("unknown probability requires unknown expected value")
            return self
        calculated = (
            self.probability_win * self.reward
            - (1.0 - self.probability_win) * self.risk
            - self.trading_cost
        )
        calculated_r = calculated / self.risk
        if self.expected_value is None or abs(self.expected_value - calculated) > 1e-12:
            raise ValueError("expected_value does not match Trader's Equation")
        if self.expected_value_r is None or abs(self.expected_value_r - calculated_r) > 1e-12:
            raise ValueError("expected_value_r does not match Trader's Equation")
        return self


def calculate_expected_value(
    *,
    probability_win: float | None,
    reward: float,
    risk: float,
    trading_cost: float = 0.0,
) -> ExpectedValue:
    """Calculate EV only when a statistics-derived win probability is supplied."""
    if probability_win is None:
        return ExpectedValue(
            probability_win=None,
            reward=reward,
            risk=risk,
            trading_cost=trading_cost,
            expected_value=None,
            expected_value_r=None,
        )
    expected_value = probability_win * reward - (1.0 - probability_win) * risk - trading_cost
    return ExpectedValue(
        probability_win=probability_win,
        reward=reward,
        risk=risk,
        trading_cost=trading_cost,
        expected_value=expected_value,
        expected_value_r=expected_value / risk,
    )
