"""Node implementations for the Automaton graph."""

import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState, CodeEdit, CritiqueResult, Plan, TestResult
from tools import list_dir, read_file, run_command, write_file


CONTEXT_FILE_SUFFIXES = {".py", ".toml", ".md"}
MAX_CONTEXT_FILES = 8
PLANNER_MODEL = os.getenv("AUTOMATON_PLANNER_MODEL", "gemini-2.5-flash-lite")
CODER_MODEL = os.getenv("AUTOMATON_CODER_MODEL", "gemini-2.5-flash-lite")
CRITIC_MODEL = os.getenv("AUTOMATON_CRITIC_MODEL", "gemini-2.5-flash-lite")


def _google_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, temperature=0.0)


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
    print("-> Planner node")
    working_dir = state["working_dir"]
    file_tree = list_dir.invoke({"path": ".", "working_dir": working_dir, "depth": 2})
    code_context = _read_code_context(file_tree, working_dir)
    critique_context = state.get("critique")

    planner_llm = _google_llm(PLANNER_MODEL).with_structured_output(Plan)

    plan = planner_llm.invoke(
        (
            "Create a concise implementation plan for this coding task.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"Previous critique, if any:\n{critique_context}\n\n"
            f"File tree:\n{file_tree}\n\n"
            f"Code context:\n{code_context}"
        )
    )

    return {
        "file_tree": file_tree,
        "code_context": code_context,
        "plan": plan,
        "next": "coder",
        "trajectory": ["planner created plan"],
    }


def coder(state: AgentState) -> dict[str, object]:
    """Produce and apply a structured full-file code edit."""
    print("-> Coder node")
    coder_llm = _google_llm(CODER_MODEL).with_structured_output(CodeEdit)

    test_result = state.get("test_result")
    failure_context = test_result.failure_summary if test_result is not None else "No tests run yet."
    critique_context = state.get("critique")

    code_edit = coder_llm.invoke(
        (
            "Return exactly one full-file edit that addresses the task and failing tests.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Critique:\n{critique_context}\n\n"
            f"Code context:\n{state['code_context']}\n\n"
            f"Latest test failure:\n{failure_context}"
        )
    )
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
            "trajectory": [f"coder failed to write {code_edit.file_path}: {write_result}"],
        }

    return {
        "last_edit": code_edit,
        "last_error": None,
        "next": "executor",
        "trajectory": [f"coder edited {code_edit.file_path}"],
    }


def executor(state: AgentState) -> dict[str, object]:
    """Run pytest and parse the result."""
    print("-> Executor node")
    if state["status"] == "failed":
        return {
            "next": "critic",
            "trajectory": ["executor skipped because status=failed"],
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
        "trajectory": [f"executor passed={test_result.passed} failed={test_result.failed_count}"],
    }


def critic(state: AgentState) -> dict[str, object]:
    """Critique the latest attempt and decide the next route."""
    print("-> Critic node")
    iteration = state.get("iteration", 0) + 1

    test_result = state.get("test_result")
    last_edit = state.get("last_edit")

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
        critic_llm = _google_llm(CRITIC_MODEL).with_structured_output(CritiqueResult)
        critique = critic_llm.invoke(
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
            )
        )

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
            f"critic iteration={iteration} verdict={critique.verdict} summary={critique.summary}"
        ],
    }
