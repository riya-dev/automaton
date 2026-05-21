"""Run Automaton against the local benchmark tasks."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from agent import graph


load_dotenv()


class TaskMetadata(BaseModel):
    """Metadata describing one benchmark task."""

    id: str
    prompt: str
    category: str
    difficulty: str


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
        "eval_result": None,
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def load_tasks(tasks_root: Path) -> list[tuple[TaskMetadata, Path]]:
    tasks: list[tuple[TaskMetadata, Path]] = []

    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        metadata_path = task_dir / "task.json"
        if not metadata_path.exists():
            continue

        metadata = TaskMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        tasks.append((metadata, task_dir))

    return tasks


def run_task(metadata: TaskMetadata, source_dir: Path, temp_root: Path) -> dict[str, Any]:
    work_dir = temp_root / metadata.id
    shutil.copytree(
        source_dir,
        work_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )

    started_at = time.perf_counter()
    try:
        result = graph.invoke(_initial_state(metadata.prompt, work_dir))
        duration = time.perf_counter() - started_at
        eval_result = result.get("eval_result")
        critique = result.get("critique")

        return {
            "task_id": metadata.id,
            "category": metadata.category,
            "difficulty": metadata.difficulty,
            "success": bool(getattr(eval_result, "success", False)),
            "status": getattr(eval_result, "final_status", result.get("status")),
            "iterations": getattr(eval_result, "iterations_used", result.get("iteration", 0)),
            "duration_seconds": round(duration, 2),
            "critique_verdict": getattr(critique, "verdict", None),
            "trajectory": _jsonable(result.get("trajectory", [])),
            "critique": _jsonable(critique),
            "eval_result": _jsonable(eval_result),
        }
    except Exception as error:
        duration = time.perf_counter() - started_at
        return {
            "task_id": metadata.id,
            "category": metadata.category,
            "difficulty": metadata.difficulty,
            "success": False,
            "status": "error",
            "iterations": 0,
            "duration_seconds": round(duration, 2),
            "critique_verdict": None,
            "error": str(error),
            "trajectory": [],
            "critique": None,
            "eval_result": None,
        }


def summarize_results(results: list[dict[str, Any]], created_at: str) -> dict[str, Any]:
    total_tasks = len(results)
    passed = sum(1 for result in results if result["success"])
    failed = total_tasks - passed
    total_iterations = sum(result["iterations"] for result in results)
    total_duration = sum(result["duration_seconds"] for result in results)

    return {
        "created_at": created_at,
        "total_tasks": total_tasks,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total_tasks, 3) if total_tasks else 0.0,
        "average_iterations": round(total_iterations / total_tasks, 2)
        if total_tasks
        else 0.0,
        "average_duration_seconds": round(total_duration / total_tasks, 2)
        if total_tasks
        else 0.0,
    }


def metrics_for_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "category": result["category"],
        "difficulty": result["difficulty"],
        "success": result["success"],
        "status": result["status"],
        "iterations": result["iterations"],
        "duration_seconds": result["duration_seconds"],
        "critique_verdict": result["critique_verdict"],
    }


def report_for_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "category": result["category"],
        "difficulty": result["difficulty"],
        "trajectory": result["trajectory"],
        "critique": result["critique"],
        "eval_result": result["eval_result"],
        **({"error": result["error"]} if "error" in result else {}),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Summary:")
    print(f"- total tasks: {summary['total_tasks']}")
    print(f"- passed: {summary['passed']}")
    print(f"- failed: {summary['failed']}")
    print(f"- pass rate: {summary['pass_rate']:.1%}")
    print(f"- average iterations: {summary['average_iterations']}")
    print(f"- average duration: {summary['average_duration_seconds']}s")
    print()


def print_table(results: list[dict[str, Any]]) -> None:
    print("| task | category | success | status | verdict | iterations | seconds |")
    print("| --- | --- | --- | --- | --- | ---: | ---: |")
    for result in results:
        print(
            "| {task_id} | {category} | {success} | {status} | "
            "{critique_verdict} | {iterations} | {duration_seconds} |".format(**result)
        )


def main() -> None:
    root = Path(__file__).parent
    tasks_root = root / "benchmarks" / "tasks"
    results_path = root / "benchmark_results.json"
    trajectory_path = root / "trajectory_report.json"
    created_at = datetime.now(UTC).isoformat()
    tasks = load_tasks(tasks_root)

    if not tasks:
        raise SystemExit(f"No task.json files found under {tasks_root}")

    results = []
    with tempfile.TemporaryDirectory(prefix="automaton-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        for metadata, task_dir in tasks:
            print(f"Running {metadata.id}...")
            results.append(run_task(metadata, task_dir, temp_root))

    print()
    summary = summarize_results(results, created_at)
    print_summary(summary)
    print_table(results)
    results_payload = {
        "created_at": created_at,
        "summary": summary,
        "results": [metrics_for_result(result) for result in results],
    }
    trajectory_payload = {
        "created_at": created_at,
        "runs": [report_for_result(result) for result in results],
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    trajectory_path.write_text(json.dumps(trajectory_payload, indent=2), encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {trajectory_path}")


if __name__ == "__main__":
    main()
