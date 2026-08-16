"""Auditable contracts for book text, knowledge chunks, and reviewed rules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from brooks_trader.models.common import DomainModel


class BookParagraph(DomainModel):
    """One ordered paragraph extracted from a licensed local source book."""

    book: str = Field(min_length=1)
    chapter: str = Field(min_length=1)
    section: str | None = None
    paragraph: int = Field(ge=1)
    text: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_document: str = Field(min_length=1)


class KnowledgeChunk(DomainModel):
    """A retrievable text unit with exact paragraph provenance."""

    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    book: str = Field(min_length=1)
    chapter: str = Field(min_length=1)
    section: str | None = None
    paragraph_start: int = Field(ge=1)
    paragraph_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_documents: list[str] = Field(min_length=1)
    source_reference: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paragraph_range(self) -> KnowledgeChunk:
        if self.paragraph_end < self.paragraph_start:
            raise ValueError("paragraph_end cannot precede paragraph_start")
        return self


class RuleStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class RuleDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class BrooksRuleSource(DomainModel):
    """Human-verifiable location in a Brooks source book."""

    book: str = Field(min_length=1)
    chapter: str = Field(min_length=1)
    section: str | None = None
    source_reference: str = Field(min_length=1)


class BrooksRule(DomainModel):
    """Documentation-level Brooks rule; never executable natural-language code."""

    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    direction: RuleDirection
    source: BrooksRuleSource
    definition: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    entry: dict[str, Any] = Field(default_factory=dict)
    stop: dict[str, Any] = Field(default_factory=dict)
    target: dict[str, Any] = Field(default_factory=dict)
    failure: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    status: RuleStatus = RuleStatus.CANDIDATE
    implementation: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @field_validator("reviewed_at")
    @classmethod
    def normalize_review_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_review_gate(self) -> BrooksRule:
        if self.status == RuleStatus.APPROVED:
            if not self.reviewed_by or self.reviewed_at is None:
                raise ValueError("approved rules require reviewed_by and reviewed_at")
            if self.source.chapter.casefold() == "unverified" or (
                "verification required" in self.source.source_reference.casefold()
            ):
                raise ValueError("approved rules require a verified source reference")
        return self

    @property
    def is_strategy_eligible(self) -> bool:
        """Only a human-approved rule may be considered by production code."""
        return self.status == RuleStatus.APPROVED
