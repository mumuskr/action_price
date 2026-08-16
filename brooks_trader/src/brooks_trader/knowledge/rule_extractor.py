"""Human-review gate and YAML persistence for Brooks rule candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from brooks_trader.models import BrooksRule, RuleStatus


def load_rule(path: str | Path) -> BrooksRule:
    """Load one declarative rule as metadata, never as executable Python."""
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"rule YAML must contain one mapping: {source}")
    try:
        return BrooksRule.model_validate(raw)
    except ValueError as error:
        raise ValueError(f"invalid Brooks rule: {source}") from error


def load_rule_library(root: str | Path) -> list[BrooksRule]:
    """Load every rule YAML recursively and reject duplicate names."""
    directory = Path(root).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"rule library directory does not exist: {directory}")
    rules = [load_rule(path) for path in sorted(directory.rglob("*.yaml"))]
    duplicates = _duplicates(rule.name for rule in rules)
    if duplicates:
        raise ValueError(f"duplicate Brooks rule names: {sorted(duplicates)}")
    return rules


def approved_rules(rules: Sequence[BrooksRule]) -> list[BrooksRule]:
    """Return only rules that have passed the explicit human approval gate."""
    validated = [BrooksRule.model_validate(rule.model_dump()) for rule in rules]
    return [rule for rule in validated if rule.status == RuleStatus.APPROVED]


def candidate_rule(values: Mapping[str, Any]) -> BrooksRule:
    """Validate extraction output while forcing it to remain a candidate."""
    candidate = dict(values)
    candidate["status"] = RuleStatus.CANDIDATE.value
    candidate["reviewed_by"] = None
    candidate["reviewed_at"] = None
    return BrooksRule.model_validate(candidate)


def write_rule(rule: BrooksRule, path: str | Path) -> Path:
    """Atomically persist a validated rule for later human review."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    payload = rule.model_dump(mode="json", exclude_none=True)
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    temporary.replace(destination)
    return destination


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = value.strip().casefold()
        if normalized in seen:
            duplicates.add(value)
        seen.add(normalized)
    return duplicates
