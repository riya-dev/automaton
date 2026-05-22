"""Run Automaton against the local benchmark tasks."""

from __future__ import annotations

import json
import hashlib
import argparse
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import BaseModel

from agent import graph
from benchmark_utils import strategy_slug as make_strategy_slug


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


def load_task(task_dir: Path) -> TaskMetadata:
    metadata_path = task_dir / "task.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing task metadata: {metadata_path}")

    return TaskMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))


def test_file_hashes(task_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(task_dir.glob("test_*.py")):
        relative_path = path.relative_to(task_dir).as_posix()
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _trace_config(
    metadata: TaskMetadata,
    source_dir: Path,
    work_dir: Path,
    run_source: str,
    strategy: str,
) -> dict[str, Any]:
    return {
        "run_name": f"automaton:{metadata.id}",
        "tags": [
            "automaton",
            "benchmark",
            run_source,
            f"strategy:{strategy}",
            metadata.category,
            metadata.difficulty,
        ],
        "metadata": {
            "task_id": metadata.id,
            "category": metadata.category,
            "difficulty": metadata.difficulty,
            "source_dir": str(source_dir),
            "working_dir": str(work_dir),
            "max_iterations": 6,
            "run_source": run_source,
            "strategy": strategy,
        },
    }


def run_task(
    metadata: TaskMetadata,
    source_dir: Path,
    temp_root: Path,
    run_source: str = "eval_harness",
    strategy: str = "structured",
) -> dict[str, Any]:
    work_dir = temp_root / metadata.id
    shutil.copytree(
        source_dir,
        work_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    initial_test_hashes = test_file_hashes(work_dir)

    started_at = time.perf_counter()
    try:
        result = graph.invoke(
            _initial_state(metadata.prompt, work_dir),
            config=_trace_config(metadata, source_dir, work_dir, run_source, strategy),
        )
        duration = time.perf_counter() - started_at
        eval_result = result.get("eval_result")
        critique = result.get("critique")
        final_test_hashes = test_file_hashes(work_dir)
        changed_or_deleted_tests = [
            path
            for path, initial_hash in initial_test_hashes.items()
            if final_test_hashes.get(path) != initial_hash
        ]
        new_tests = [
            path for path in final_test_hashes if path not in initial_test_hashes
        ]
        changed_tests = sorted(changed_or_deleted_tests + new_tests)
        tests_unchanged = not changed_tests

        return {
            "task_id": metadata.id,
            "category": metadata.category,
            "difficulty": metadata.difficulty,
            "strategy": strategy,
            "success": bool(getattr(eval_result, "success", False)) and tests_unchanged,
            "status": getattr(eval_result, "final_status", result.get("status")),
            "iterations": getattr(eval_result, "iterations_used", result.get("iteration", 0)),
            "input_tokens": getattr(eval_result, "input_tokens", 0),
            "output_tokens": getattr(eval_result, "output_tokens", 0),
            "thinking_tokens": getattr(eval_result, "thinking_tokens", 0),
            "cost_usd": getattr(eval_result, "cost_usd", 0.0),
            "duration_seconds": round(duration, 2),
            "critique_verdict": getattr(critique, "verdict", None),
            "tests_unchanged": tests_unchanged,
            "changed_tests": changed_tests,
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
            "strategy": strategy,
            "success": False,
            "status": "error",
            "iterations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cost_usd": 0.0,
            "duration_seconds": round(duration, 2),
            "critique_verdict": None,
            "tests_unchanged": False,
            "changed_tests": [],
            "error": str(error),
            "trajectory": [],
            "critique": None,
            "eval_result": None,
        }


def summarize_results(
    results: list[dict[str, Any]],
    created_at: str,
    strategy: str,
) -> dict[str, Any]:
    total_tasks = len(results)
    passed = sum(1 for result in results if result["success"])
    failed = total_tasks - passed
    total_iterations = sum(result["iterations"] for result in results)
    total_duration = sum(result["duration_seconds"] for result in results)
    total_input_tokens = sum(result["input_tokens"] for result in results)
    total_output_tokens = sum(result["output_tokens"] for result in results)
    total_thinking_tokens = sum(result["thinking_tokens"] for result in results)
    total_cost_usd = sum(result["cost_usd"] for result in results)

    return {
        "created_at": created_at,
        "strategy": strategy,
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
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_thinking_tokens": total_thinking_tokens,
        "total_cost_usd": round(total_cost_usd, 8),
        "average_cost_usd": round(total_cost_usd / total_tasks, 8) if total_tasks else 0.0,
    }


def metrics_for_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "category": result["category"],
        "difficulty": result["difficulty"],
        "strategy": result["strategy"],
        "success": result["success"],
        "status": result["status"],
        "iterations": result["iterations"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "thinking_tokens": result["thinking_tokens"],
        "cost_usd": result["cost_usd"],
        "duration_seconds": result["duration_seconds"],
        "critique_verdict": result["critique_verdict"],
        "tests_unchanged": result["tests_unchanged"],
        "changed_tests": result["changed_tests"],
    }


def report_for_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "category": result["category"],
        "difficulty": result["difficulty"],
        "strategy": result["strategy"],
        "trajectory": result["trajectory"],
        "critique": result["critique"],
        "eval_result": result["eval_result"],
        "tests_unchanged": result["tests_unchanged"],
        "changed_tests": result["changed_tests"],
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
    print(f"- total tokens: {summary['total_input_tokens']} in / {summary['total_output_tokens']} out / {summary['total_thinking_tokens']} thinking")
    print(f"- total cost: ${summary['total_cost_usd']:.8f}")
    print(f"- average cost: ${summary['average_cost_usd']:.8f}")
    print()


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        "| task | category | success | status | verdict | tests | iterations | "
        "seconds | cost_usd |"
    )
    print("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for result in results:
        print(
            "| {task_id} | {category} | {success} | {status} | "
            "{critique_verdict} | {tests_unchanged} | {iterations} | "
            "{duration_seconds} | {cost_usd:.8f} |".format(**result)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Automaton against benchmark tasks.")
    parser.add_argument(
        "--strategy",
        default="structured",
        help=(
            "Experiment label to attach to LangSmith traces and local reports "
            "(for example: structured or react)."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    tasks_root = root / "benchmarks" / "tasks"
    results_path = root / "benchmark_results.json"
    trajectory_path = root / "trajectory_report.json"
    created_at = datetime.now(UTC).isoformat()
    history_id = created_at.replace(":", "-")
    strategy = args.strategy.strip() or "structured"
    strategy_slug = make_strategy_slug(strategy)
    runs_root = root / "benchmark_runs"
    latest_results_path = runs_root / "latest_results.json"
    latest_trajectory_path = runs_root / "latest_trajectory.json"
    latest_strategy_results_path = runs_root / f"latest_{strategy_slug}_results.json"
    latest_strategy_trajectory_path = runs_root / f"latest_{strategy_slug}_trajectory.json"
    history_root = runs_root / "history"
    history_results_path = history_root / f"{history_id}_{strategy_slug}_results.json"
    history_trajectory_path = history_root / f"{history_id}_{strategy_slug}_trajectory.json"
    tasks = load_tasks(tasks_root)

    if not tasks:
        raise SystemExit(f"No task.json files found under {tasks_root}")

    results = []
    with tempfile.TemporaryDirectory(prefix="automaton-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        for metadata, task_dir in tasks:
            print(f"Running {metadata.id}...")
            results.append(run_task(metadata, task_dir, temp_root, strategy=strategy))

    print()
    summary = summarize_results(results, created_at, strategy)
    print_summary(summary)
    print_table(results)
    results_payload = {
        "created_at": created_at,
        "strategy": strategy,
        "summary": summary,
        "results": [metrics_for_result(result) for result in results],
    }
    trajectory_payload = {
        "created_at": created_at,
        "strategy": strategy,
        "runs": [report_for_result(result) for result in results],
    }
    runs_root.mkdir(exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    trajectory_path.write_text(json.dumps(trajectory_payload, indent=2), encoding="utf-8")
    latest_results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    latest_trajectory_path.write_text(json.dumps(trajectory_payload, indent=2), encoding="utf-8")
    latest_strategy_results_path.write_text(
        json.dumps(results_payload, indent=2),
        encoding="utf-8",
    )
    latest_strategy_trajectory_path.write_text(
        json.dumps(trajectory_payload, indent=2),
        encoding="utf-8",
    )
    history_results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    history_trajectory_path.write_text(
        json.dumps(trajectory_payload, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {results_path}")
    print(f"Wrote {trajectory_path}")
    print(f"Wrote {latest_results_path}")
    print(f"Wrote {latest_trajectory_path}")
    print(f"Wrote {latest_strategy_results_path}")
    print(f"Wrote {latest_strategy_trajectory_path}")
    print(f"Wrote {history_results_path}")
    print(f"Wrote {history_trajectory_path}")
    wait_for_all_tracers()


if __name__ == "__main__":
    main()
