"""Streamlit UI for running Automaton benchmark tasks."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "benchmarks" / "tasks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness import load_tasks, run_task


def _load_task_options() -> list[tuple[str, Path, Any]]:
    tasks = load_tasks(TASKS_ROOT)
    return [
        (
            f"{metadata.id} - {metadata.category} ({metadata.difficulty})",
            task_dir,
            metadata,
        )
        for metadata, task_dir in tasks
    ]


def _trajectory_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in result.get("trajectory") or []:
        rows.append(
            {
                "node": step.get("node"),
                "summary": step.get("summary"),
                "decision": step.get("decision"),
                "latency_seconds": step.get("latency_seconds"),
                "model": step.get("model"),
                "input_tokens": step.get("input_tokens", 0),
                "output_tokens": step.get("output_tokens", 0),
                "cost_usd": step.get("cost_usd", 0.0),
            }
        )
    return rows


def _show_result(result: dict[str, Any]) -> None:
    success = bool(result.get("success"))
    status = result.get("status", "unknown")
    eval_result = result.get("eval_result") or {}
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    cost_usd = result.get("cost_usd", 0.0)

    if success:
        st.success(f"{result['task_id']} passed")
    else:
        st.error(f"{result.get('task_id', 'task')} ended with status={status}")

    metric_cols = st.columns(7)
    metric_cols[0].metric("Status", status)
    metric_cols[1].metric("Iterations", result.get("iterations", 0))
    metric_cols[2].metric("Seconds", result.get("duration_seconds", 0))
    metric_cols[3].metric("Verdict", result.get("critique_verdict") or "none")
    metric_cols[4].metric(
        "Tests", "unchanged" if result.get("tests_unchanged") else "changed"
    )
    metric_cols[5].metric("Estimated Cost", f"${cost_usd:.6f}")
    metric_cols[6].metric("Total Tokens", total_tokens)

    changed_tests = result.get("changed_tests") or []
    if changed_tests:
        st.warning(", ".join(changed_tests))

    st.subheader("Trajectory")
    rows = _trajectory_rows(result)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No trajectory recorded.")

    critique = result.get("critique")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.subheader("Critique")
        if critique:
            st.json(critique)
        else:
            st.caption("No critique recorded.")

    with detail_cols[1]:
        st.subheader("Evaluation")
        if eval_result:
            st.json(eval_result)
        else:
            st.caption("No evaluation recorded.")

    with st.expander("Raw result"):
        st.code(json.dumps(result, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Automaton", layout="wide")
    st.title("Automaton")

    task_options = _load_task_options()
    if not task_options:
        st.error(f"No benchmark tasks found under {TASKS_ROOT}")
        return

    with st.sidebar:
        selected_label = st.selectbox(
            "Benchmark",
            [label for label, _, _ in task_options],
            index=0,
        )
        selected = next(item for item in task_options if item[0] == selected_label)
        _, task_dir, metadata = selected

        st.text_area("Prompt", metadata.prompt, height=140, disabled=True)
        run_selected = st.button("Run", type="primary", use_container_width=True)

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    header_cols = st.columns([2, 1, 1])
    header_cols[0].subheader(metadata.id)
    header_cols[1].metric("Category", metadata.category)
    header_cols[2].metric("Difficulty", metadata.difficulty)

    if run_selected:
        with st.spinner(f"Running {metadata.id}"):
            with tempfile.TemporaryDirectory(prefix="automaton-ui-") as temp_dir:
                st.session_state.last_result = run_task(
                    metadata,
                    task_dir,
                    Path(temp_dir),
                    run_source="streamlit",
                )

    if st.session_state.last_result is None:
        st.info("Select a benchmark task and run it.")
        return

    _show_result(st.session_state.last_result)


if __name__ == "__main__":
    main()
