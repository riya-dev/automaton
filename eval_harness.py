"""Run Automaton against the local benchmark tasks."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent import graph


load_dotenv()


TASKS = {
    "task_001": "Fix next_number so it returns the integer after the input value.",
    "task_002": (
        "Fix last_index so it returns the final valid zero-based index for a list, "
        "or -1 for an empty list."
    ),
    "task_003": "Fix most_common_word so it returns the most frequently occurring word.",
    "task_004": "Fix parse_count so it parses and returns an integer count from text.",
    "task_005": (
        "Fix can_edit_document so document owners or admins can edit, while "
        "unprivileged users cannot."
    ),
}


def _initial_state(task: str, working_dir: Path) -> dict[str, Any]:
    return {
        "task": task,
        "next": None,
        "iteration": 0,
        "max_iterations": 6,
        "messages": [],
        "working_dir": str(working_dir),
        "file_tree": "",
        "code_context": "",
        "plan": None,
        "last_edit": None,
        "test_result": None,
        "last_error": None,
        "status": "running",
        "critique": None,
        "trajectory": [],
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run_task(task_id: str, task: str, source_dir: Path, temp_root: Path) -> dict[str, Any]:
    work_dir = temp_root / task_id
    shutil.copytree(
        source_dir,
        work_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )

    started_at = time.perf_counter()
    try:
        result = graph.invoke(_initial_state(task, work_dir))
        duration = time.perf_counter() - started_at
        test_result = result.get("test_result")
        success = result.get("status") == "passed" and bool(
            getattr(test_result, "passed", False)
        )

        return {
            "task_id": task_id,
            "success": success,
            "status": result.get("status"),
            "iterations": result.get("iteration", 0),
            "duration_seconds": round(duration, 2),
            "trajectory": result.get("trajectory", []),
            "critique": _jsonable(result.get("critique")),
        }
    except Exception as error:
        duration = time.perf_counter() - started_at
        return {
            "task_id": task_id,
            "success": False,
            "status": "error",
            "iterations": 0,
            "duration_seconds": round(duration, 2),
            "error": str(error),
            "trajectory": [],
            "critique": None,
        }


def print_table(results: list[dict[str, Any]]) -> None:
    print("| task | success | status | iterations | seconds |")
    print("| --- | --- | --- | ---: | ---: |")
    for result in results:
        print(
            "| {task_id} | {success} | {status} | {iterations} | {duration_seconds} |".format(
                **result
            )
        )


def main() -> None:
    root = Path(__file__).parent
    tasks_root = root / "benchmarks" / "tasks"
    results_path = root / "benchmark_results.json"

    results = []
    with tempfile.TemporaryDirectory(prefix="automaton-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        for task_id, task in TASKS.items():
            print(f"Running {task_id}...")
            results.append(run_task(task_id, task, tasks_root / task_id, temp_root))

    print()
    print_table(results)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
