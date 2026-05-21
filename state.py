"""Shared state definitions for Automaton."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class PlanStep(BaseModel):
    """One concrete step in the agent's implementation plan."""

    step_id: int
    description: str
    target_file: str | None


class Plan(BaseModel):
    """Structured plan produced by the planner node."""

    goal_summary: str
    steps: list[PlanStep]


class CodeEdit(BaseModel):
    """Full-file edit produced by the coder node."""

    file_path: str
    content: str


class TestResult(BaseModel):
    """Parsed test execution result produced by the executor node."""

    passed: bool
    passed_count: int
    failed_count: int
    failure_summary: str
    raw_output: str


class CritiqueResult(BaseModel):
    """Structured judgment produced by the critic node."""

    summary: str
    issues: list[str]
    confidence: float
    verdict: Literal["continue", "replan", "done", "give_up"]


class TrajectoryStep(BaseModel):
    """One observable step in an agent run."""

    node: str
    summary: str
    decision: str | None = None
    latency_seconds: float | None = None


class EvalResult(BaseModel):
    """Final deterministic evaluation for one agent run."""

    success: bool
    final_status: str
    iterations_used: int
    trajectory_efficiency: float
    summary: str


class AgentState(TypedDict):
    """State passed between graph nodes."""

    task: str
    next: str | None
    iteration: int
    max_iterations: int
    messages: Annotated[list[BaseMessage], operator.add]
    working_dir: str
    file_tree: str
    code_context: str
    plan: Plan | None
    last_edit: CodeEdit | None
    test_result: TestResult | None
    last_error: str | None
    status: Literal["running", "passed", "failed", "max_iter_reached"]
    critique: CritiqueResult | None
    trajectory: Annotated[list[TrajectoryStep], operator.add]
    eval_result: EvalResult | None
