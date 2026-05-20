"""Stub node implementations for the Day 1 graph."""

from state import AgentState


def planner(_state: AgentState) -> dict[str, str]:
    """Stub planner node."""
    print("-> Planner node")
    return {"next": "coder"}


def coder(_state: AgentState) -> dict[str, str]:
    """Stub coder node."""
    print("-> Coder node")
    return {"next": "executor"}


def executor(_state: AgentState) -> dict[str, str]:
    """Stub executor node."""
    print("-> Executor node")
    return {"next": "critic"}


def critic(state: AgentState) -> dict[str, int | str]:
    """Stub critic node."""
    print("-> Critic node")
    iteration = state.get("iteration", 0) + 1
    return {"iteration": iteration, "next": "end"}
