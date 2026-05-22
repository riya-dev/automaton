"""Run one benchmark task through Automaton."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers

from eval_harness import load_task, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Automaton benchmark task.")
    parser.add_argument("task_dir", type=Path, help="Path to a benchmark task directory.")
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    metadata = load_task(task_dir)

    with tempfile.TemporaryDirectory(prefix="automaton-task-") as temp_dir:
        result = run_task(metadata, task_dir, Path(temp_dir))

    print(json.dumps(result, indent=2))
    wait_for_all_tracers()


if __name__ == "__main__":
    main()
