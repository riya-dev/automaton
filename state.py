"""Shared state definitions for Automaton."""

from typing import TypedDict


class AgentState(TypedDict):
    """Minimal Day 1 state passed between graph nodes."""

    task: str
    next: str | None
    iteration: int
    max_iterations: int
