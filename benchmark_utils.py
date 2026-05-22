"""Shared helpers for benchmark scripts."""

from __future__ import annotations


def strategy_slug(strategy: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in strategy.strip().lower()
    ).strip("-")
    return slug or "structured"
