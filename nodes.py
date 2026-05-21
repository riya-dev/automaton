"""Node implementations for the Automaton graph."""

import re

from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState, CodeEdit, Plan, TestResult
from tools import list_dir, read_file, run_command, write_file


CONTEXT_FILE_SUFFIXES = {".py", ".toml", ".md"}
MAX_CONTEXT_FILES = 8


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

    planner_llm = _google_llm("gemini-2.5-pro").with_structured_output(Plan)

    plan = planner_llm.invoke(
        (
            "Create a concise implementation plan for this coding task.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"File tree:\n{file_tree}\n\n"
            f"Code context:\n{code_context}"
        )
    )

    return {
        "file_tree": file_tree,
        "code_context": code_context,
        "plan": plan,
        "next": "coder",
    }


def coder(state: AgentState) -> dict[str, object]:
    """Produce and apply a structured full-file code edit."""
    print("-> Coder node")
    coder_llm = _google_llm("gemini-2.5-flash").with_structured_output(CodeEdit)

    test_result = state.get("test_result")
    failure_context = test_result.failure_summary if test_result is not None else "No tests run yet."

    code_edit = coder_llm.invoke(
        (
            "Return exactly one full-file edit that addresses the task and failing tests.\n\n"
            f"Task:\n{state['task']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
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
        return {"last_error": write_result, "status": "failed", "next": "executor"}

    return {
        "last_edit": code_edit,
        "last_error": None,
        "next": "executor",
    }


def executor(state: AgentState) -> dict[str, object]:
    """Run pytest and parse the result."""
    print("-> Executor node")
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
    }


def critic(state: AgentState) -> dict[str, int | str]:
    """Update loop status after a test execution."""
    print("-> Critic node")
    iteration = state.get("iteration", 0) + 1

    if state["status"] == "passed":
        return {"iteration": iteration, "next": "end"}

    if state["status"] == "failed":
        return {"iteration": iteration, "next": "end"}

    if iteration >= state["max_iterations"]:
        return {
            "iteration": iteration,
            "status": "max_iter_reached",
            "next": "end",
        }

    return {
        "iteration": iteration,
        "status": "running",
        "next": "coder",
    }
