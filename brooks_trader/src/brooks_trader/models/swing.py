"""Confirmed swing-point contract with explicit detection latency."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import DomainModel


class SwingType(StrEnum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingPoint(DomainModel):
    """A swing whose occurrence and confirmation are recorded separately."""

    index: int = Field(ge=0)
    swing_time: datetime
    price: float = Field(gt=0)
    type: SwingType
    confirmed_at: int = Field(ge=0)
    confirmation_time: datetime

    @field_validator("swing_time", "confirmation_time")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("swing timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_confirmation(self) -> "SwingPoint":
        if self.confirmed_at < self.index:
            raise ValueError("confirmed_at cannot precede the swing index")
        if self.confirmation_time < self.swing_time:
            raise ValueError("confirmation_time cannot precede swing_time")
        return self
