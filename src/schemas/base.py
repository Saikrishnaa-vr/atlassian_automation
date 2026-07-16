"""Shared schema primitives for domain models.

This module contains only functionality that is truly shared across the domain
schema modules. It intentionally excludes domain-specific enums and status
types, which remain in their respective modules.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base class for domain schema contracts.

    All domain models are immutable and reject unknown fields so contract
    mismatches fail loudly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def require_non_blank(value: str, *, field: str) -> str:
    """Return trimmed text, rejecting empty or whitespace-only values."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be blank")
    return cleaned


def clean_unique_tokens(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    """Trim token lists and reject blanks/duplicates while preserving order."""
    cleaned = tuple(value.strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{field} must not contain blank entries")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field} must not contain duplicate entries")
    return cleaned
