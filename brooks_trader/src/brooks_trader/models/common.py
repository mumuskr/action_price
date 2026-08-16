"""Common enums and Pydantic behavior for domain models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Direction(StrEnum):
    """Directional intent, independent of an order's execution type."""

    LONG = "LONG"
    SHORT = "SHORT"


class DomainModel(BaseModel):
    """Immutable base that rejects undeclared fields in research artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)
