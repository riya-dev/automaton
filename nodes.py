"""Node implementations for the Automaton graph."""

import os
import re
import time
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from state import (
    AgentState,
    CodeEdit,
    CritiqueResult,
    EvalResult,
    Plan,
    TestResult,
    TrajectoryStep,
)
from tools import list_dir, read_file, run_command, write_file


CONTEXT_FILE_SUFFIXES = {".py", ".toml", ".md"}
MAX_CONTEXT_FILES = 8
PLANNER_MODEL = os.getenv("AUTOMATON_PLANNER_MODEL", "gemini-2.5-flash-lite")
CODER_MODEL = os.getenv("AUTOMATON_CODER_MODEL", "gemini-2.5-flash-lite")
CRITIC_MODEL = os.getenv("AUTOMATON_CRITIC_MODEL", "gemini-2.5-flash-lite")
LOGGER = logging.getLogger(__name__)
GEMINI_FLASH_LITE_INPUT_PER_1M = float(
    os.getenv("AUTOMATON_GEMINI_FLASH_LITE_INPUT_PER_1M", "0.10")
)
GEMINI_FLASH_LITE_OUTPUT_PER_1M = float(
    os.getenv("AUTOMATON_GEMINI_FLASH_LITE_OUTPUT_PER_1M", "0.40")
)


def _google_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, temperature=0.0)


def _model_rates(model: str | None) -> tuple[float, float]:
    if model == "gemini-2.5-flash-lite":
        return GEMINI_FLASH_LITE_INPUT_PER_1M, GEMINI_FLASH_LITE_OUTPUT_PER_1M
    if model is not None:
        LOGGER.warning("No pricing configured for model %r; cost_usd will be 0.0", model)
    return 0.0, 0.0


def _cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _model_rates(model)
    return ((input_tokens * input_rate) + (output_tokens * output_rate)) / 1_000_000


def _metadata_value(metadata: Any, key: str) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _usage_tokens(raw_response: Any) -> tuple[int, int]:
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if not usage_metadata:
        response_metadata = getattr(raw_response, "response_metadata", None) or {}
        usage_metadata = (
            response_metadata.get("usage_metadata")
            or response_metadata.get("token_usage")
        )
    if not usage_metadata:
        return 0, 0

    input_tokens = _metadata_value(usage_metadata, "input_tokens")
    output_tokens = _metadata_value(usage_metadata, "output_tokens")

    if input_tokens is None:
        input_tokens = _metadata_value(usage_metadata, "prompt_tokens")
    if output_tokens is None:
        output_tokens = _metadata_value(usage_metadata, "completion_tokens")

    try:
        input_count = int(input_tokens or 0)
        output_count = int(output_tokens or 0)
    except (TypeError, ValueError):
        return 0, 0

    return max(input_count, 0), max(output_count, 0)


def _usage_for_response(response: dict[str, Any], model: str) -> dict[str, Any]:
    input_tokens, output_tokens = _usage_tokens(response.get("raw"))
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": _cost_usd(model, input_tokens, output_tokens),
    }


def _step(
    node: str,
    summary: str,
    decision: str | None = None,
    started_at: float | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> TrajectoryStep:
    latency = None
    if started_at is not None:
        latency = round(time.perf_counter() - started_at, 3)

    return TrajectoryStep(
        node=node,
        summary=summary,
        decision=decision,
        latency_seconds=latency,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 8),
    )


def _read_code_context(file_tree: str, working_dir: str) -> str:
    context_parts: list[str] = []

    for line in file_tree.splitlines():
        path = line.rstrip("/")
        if not any(path.endswith(suffix) for suffix in CONTEXT_FILE_SUFFIXES):
            continue

        content = read_file.invoke({"path": path, "working_dir": working_dir})
        if content.startswith("Error reading"):
            continue

        context_parts.append(f"--- {path} ---\n{content}")
        if len(context_parts) >= MAX_CONTEXT_FILES:
            break

    return "\n\n".join(context_parts)


def _extract_exit_code(command_output: str) -> int | None:
    match = re.search(r"^EXIT_CODE:\s*(-?\d+)", command_output, re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1))


def _parse_pytest_result(command_output: str) -> TestResult:
    exit_code = _extract_exit_code(command_output)
    passed = exit_code == 0
    passed_count = 0
    failed_count = 0

    passed_match = re.search(r"(\d+)\s+passed", command_output)
    failed_match = re.search(r"(\d+)\s+failed", command_output)

    if passed_match is not None:
        passed_count = int(passed_match.group(1))
    if failed_match is not None:
        failed_count = int(failed_match.group(1))

    failure_summary = command_output[-4000:] if not passed else ""

    return TestResult(
        passed=passed,
        passed_count=passed_count,
        failed_count=failed_count,
        failure_summary=failure_summary,
        raw_output=command_output,
    )


def planner(state: AgentState) -> dict[str, object]:
    """Build context and produce a structured implementation plan."""
    started_at = time.perf_counter()
    LOGGER.debug("Planner node started")
    working_dir = state["working_dir"]
    file_tree = list_dir.invoke({"path": ".", "working_dir": working_dir, "depth": 2})
    code_context = _read_code_context(file_tree, working_dir)
    critique_context = state.get("critique")

    planner_llm = _google_llm(PLANNER_MODEL).with_structured_output(
        Plan, include_raw=True
    )

    response = planner_llm.invoke(
        (
            "Create a concise implementation plan for this coding task.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"Previous critique, if any:\n{critique_context}\n\n"
            f"File tree:\n{file_tree}\n\n"
            f"Code context:\n{code_context}"
        ),
        config={"run_name": "planner:model", "tags": ["planner", "model"]},
    )
    plan = response["parsed"]
    if plan is None:
        raise RuntimeError(f"Planner failed to parse LLM response: {response.get('parsing_error')}")
    usage = _usage_for_response(response, PLANNER_MODEL)

    return {
        "file_tree": file_tree,
        "code_context": code_context,
        "plan": plan,
        "next": "coder",
        "trajectory": [
            _step("planner", "created plan", "coder", started_at, **usage)
        ],
    }


def coder(state: AgentState) -> dict[str, object]:
    """Produce and apply a structured full-file code edit."""
    started_at = time.perf_counter()
    LOGGER.debug("Coder node started")
    coder_llm = _google_llm(CODER_MODEL).with_structured_output(
        CodeEdit, include_raw=True
    )

    test_result = state.get("test_result")
    failure_context = test_result.failure_summary if test_result is not None else "No tests run yet."
    critique_context = state.get("critique")

    response = coder_llm.invoke(
        (
            "Return exactly one full-file edit that addresses the task and failing tests.\n\n"
            "Do not edit test files unless the task explicitly asks you to update tests. "
            "For benchmark tasks, preserve all test_*.py files and fix implementation "
            "files instead.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Critique:\n{critique_context}\n\n"
            f"Code context:\n{state['code_context']}\n\n"
            f"Latest test failure:\n{failure_context}"
        ),
        config={"run_name": "coder:model", "tags": ["coder", "model"]},
    )
    code_edit = response["parsed"]
    if code_edit is None:
        raise RuntimeError(f"Coder failed to parse LLM response: {response.get('parsing_error')}")
    usage = _usage_for_response(response, CODER_MODEL)
    write_result = write_file.invoke(
        {
            "path": code_edit.file_path,
            "content": code_edit.content,
            "working_dir": state["working_dir"],
        }
    )

    if write_result.startswith("Error writing"):
        return {
            "last_error": write_result,
            "status": "failed",
            "next": "executor",
            "trajectory": [
                _step(
                    "coder",
                    f"failed to write {code_edit.file_path}: {write_result}",
                    "executor",
                    started_at,
                    **usage,
                )
            ],
        }

    return {
        "last_edit": code_edit,
        "last_error": None,
        "next": "executor",
        "trajectory": [
            _step(
                "coder",
                f"edited {code_edit.file_path}",
                "executor",
                started_at,
                **usage,
            )
        ],
    }


def executor(state: AgentState) -> dict[str, object]:
    """Run pytest and parse the result."""
    started_at = time.perf_counter()
    LOGGER.debug("Executor node started")
    if state["status"] == "failed":
        return {
            "next": "critic",
            "trajectory": [
                _step(
                    "executor",
                    "skipped because status=failed",
                    "critic",
                    started_at,
                )
            ],
        }

    output = run_command.invoke(
        {
            "command": "pytest --tb=short",
            "working_dir": state["working_dir"],
            "timeout": 60,
        }
    )
    test_result = _parse_pytest_result(output)
    status = "passed" if test_result.passed else "running"

    return {
        "test_result": test_result,
        "status": status,
        "last_error": None if test_result.passed else test_result.failure_summary,
        "next": "critic",
        "trajectory": [
            _step(
                "executor",
                f"passed={test_result.passed} failed={test_result.failed_count}",
                "critic",
                started_at,
            )
        ],
    }


def critic(state: AgentState) -> dict[str, object]:
    """Critique the latest attempt and decide the next route."""
    started_at = time.perf_counter()
    LOGGER.debug("Critic node started")
    iteration = state.get("iteration", 0) + 1

    test_result = state.get("test_result")
    last_edit = state.get("last_edit")
    usage = {
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }

    if state["status"] == "passed":
        critique = CritiqueResult(
            summary="Tests passed.",
            issues=[],
            confidence=1.0,
            verdict="done",
        )
    elif state["status"] == "failed":
        critique = CritiqueResult(
            summary=state.get("last_error") or "The agent hit an unrecoverable error.",
            issues=[state.get("last_error") or "Unknown failure."],
            confidence=1.0,
            verdict="give_up",
        )
    elif iteration >= state["max_iterations"]:
        critique = CritiqueResult(
            summary="Maximum iterations reached.",
            issues=["The agent ran out of allowed iterations before passing tests."],
            confidence=1.0,
            verdict="give_up",
        )
    else:
        critic_llm = _google_llm(CRITIC_MODEL).with_structured_output(
            CritiqueResult, include_raw=True
        )
        response = critic_llm.invoke(
            (
                "Critique the latest coding attempt. Decide whether the agent should "
                "continue coding from the current plan, replan, finish, or give up.\n\n"
                "Use verdict='continue' when the plan is still basically right and "
                "the coder should make another edit.\n"
                "Use verdict='replan' when the current plan appears wrong, incomplete,"
                "or aimed at the wrong cause.\n"
                "Use verdict='done' only when tests passed.\n"
                "Use verdict='give_up' only for unrecoverable errors or repeated lack of progress.\n\n"
                f"Task:\n{state['task']}\n\n"
                f"Plan:\n{state.get('plan')}\n\n"
                f"Last edit:\n{last_edit}\n\n"
                f"Test result:\n{test_result}\n\n"
                f"Failure summary:\n{state.get('last_error')}"
            ),
            config={"run_name": "critic:model", "tags": ["critic", "model"]},
        )
        critique = response["parsed"]
        if critique is None:
            raise RuntimeError(f"Critic failed to parse LLM response: {response.get('parsing_error')}")
        usage = _usage_for_response(response, CRITIC_MODEL)

    status = state["status"]
    if critique.verdict == "done":
        status = "passed"
    elif critique.verdict == "give_up" and iteration >= state["max_iterations"]:
        status = "max_iter_reached"
    elif critique.verdict == "give_up":
        status = "failed"

    return {
        "iteration": iteration,
        "critique": critique,
        "status": status,
        "next": critique.verdict,
        "trajectory": [
            _step(
                "critic",
                f"iteration={iteration} summary={critique.summary}",
                critique.verdict,
                started_at,
                **usage,
            )
        ],
    }


def evaluator(state: AgentState) -> dict[str, object]:
    """Produce a deterministic final evaluation for the run."""
    started_at = time.perf_counter()
    LOGGER.debug("Evaluator node started")
    iterations = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)
    success = state["status"] == "passed"
    trajectory = state.get("trajectory", [])
    input_tokens = sum(step.input_tokens for step in trajectory)
    output_tokens = sum(step.output_tokens for step in trajectory)
    cost_usd = sum(
        _cost_usd(step.model, step.input_tokens, step.output_tokens) for step in trajectory
    )
    eval_result = EvalResult(
        success=success,
        final_status=state["status"],
        iterations_used=iterations,
        trajectory_efficiency=iterations / max_iterations if max_iterations else 0.0,
        summary=(
            "Run passed all tests."
            if success
            else f"Run ended with status={state['status']} after {iterations} iterations."
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 8),
    )

    return {
        "eval_result": eval_result,
        "trajectory": [
            _step(
                "evaluator",
                eval_result.summary,
                "end",
                started_at,
            )
        ],
    }
