"""Compare two Automaton benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_utils import strategy_slug as make_strategy_slug


ROOT = Path(__file__).parent
RUNS_ROOT = ROOT / "benchmark_runs"


def _load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_for_strategy(strategy: str) -> Path:
    strategy_slug = make_strategy_slug(strategy)
    explicit_latest = RUNS_ROOT / f"latest_{strategy_slug}_results.json"
    if explicit_latest.exists():
        return explicit_latest

    matches = sorted((RUNS_ROOT / "history").glob(f"*_{strategy_slug}_results.json"))
    if matches:
        return matches[-1]

    raise FileNotFoundError(f"No benchmark results found for strategy={strategy!r}")


def _strategy_name(payload: dict[str, Any], fallback: str) -> str:
    return str(
        payload.get("strategy")
        or payload.get("summary", {}).get("strategy")
        or fallback
    )


def _total_tokens(summary: dict[str, Any]) -> int:
    return (
        int(summary.get("total_input_tokens", 0))
        + int(summary.get("total_output_tokens", 0))
        + int(summary.get("total_thinking_tokens", 0))
    )


def _result_tokens(result: dict[str, Any]) -> int:
    return (
        int(result.get("input_tokens", 0))
        + int(result.get("output_tokens", 0))
        + int(result.get("thinking_tokens", 0))
    )


def _format_value(metric: str, value: Any) -> str:
    if metric == "pass_rate":
        return f"{float(value):.1%}"
    if metric in {"total_cost_usd", "average_cost_usd"}:
        return f"${float(value):.8f}"
    if metric in {"average_duration_seconds", "average_iterations"}:
        return f"{float(value):.2f}"
    return str(value)


def _format_delta(metric: str, baseline: Any, candidate: Any) -> str:
    delta = float(candidate) - float(baseline)
    if metric == "pass_rate":
        return f"{delta:+.1%}"
    if metric in {"total_cost_usd", "average_cost_usd"}:
        return _format_cost_delta(delta)
    if metric in {"average_duration_seconds", "average_iterations"}:
        return f"{delta:+.2f}"
    return f"{int(delta):+d}"


def _format_cost_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else "-"
    return f"{sign}${abs(delta):.8f}"


def _summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    if key == "total_cost_usd":
        return sum(float(result.get("cost_usd", 0.0)) for result in summary["results"])
    if key == "average_cost_usd":
        total_tasks = int(summary.get("total_tasks", 0))
        if not total_tasks:
            return 0.0
        return float(_summary_value(summary, "total_cost_usd")) / total_tasks
    return 0


def _print_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
) -> None:
    baseline_summary = {**baseline["summary"], "results": baseline["results"]}
    candidate_summary = {**candidate["summary"], "results": candidate["results"]}
    metrics = [
        ("pass_rate", "Pass rate"),
        ("passed", "Passed"),
        ("failed", "Failed"),
        ("average_duration_seconds", "Avg seconds"),
        ("average_iterations", "Avg iterations"),
        ("total_cost_usd", "Total cost"),
        ("average_cost_usd", "Avg cost"),
    ]

    print(
        "| metric | {baseline} | {candidate} | delta |".format(
            baseline=baseline_name,
            candidate=candidate_name,
        )
    )
    print("| --- | ---: | ---: | ---: |")
    for key, label in metrics:
        baseline_value = _summary_value(baseline_summary, key)
        candidate_value = _summary_value(candidate_summary, key)
        print(
            "| {label} | {baseline} | {candidate} | {delta} |".format(
                label=label,
                baseline=_format_value(key, baseline_value),
                candidate=_format_value(key, candidate_value),
                delta=_format_delta(key, baseline_value, candidate_value),
            )
        )

    baseline_tokens = _total_tokens(baseline_summary)
    candidate_tokens = _total_tokens(candidate_summary)
    print(
        "| Total tokens | {baseline} | {candidate} | {delta:+d} |".format(
            baseline=baseline_tokens,
            candidate=candidate_tokens,
            delta=candidate_tokens - baseline_tokens,
        )
    )


def _result_by_task(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["task_id"]: result for result in payload["results"]}


def _print_task_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
) -> None:
    baseline_by_task = _result_by_task(baseline)
    candidate_by_task = _result_by_task(candidate)
    shared_task_ids = sorted(set(baseline_by_task) & set(candidate_by_task))

    def sort_key(task_id: str) -> tuple[bool, float, float, str]:
        baseline_result = baseline_by_task[task_id]
        candidate_result = candidate_by_task[task_id]
        success_same = baseline_result.get("success") == candidate_result.get("success")
        duration_delta = abs(
            float(candidate_result.get("duration_seconds", 0.0))
            - float(baseline_result.get("duration_seconds", 0.0))
        )
        cost_delta = abs(
            float(candidate_result.get("cost_usd", 0.0))
            - float(baseline_result.get("cost_usd", 0.0))
        )
        return (success_same, -duration_delta, -cost_delta, task_id)

    print()
    print(
        "| task | {baseline} success | {candidate} success | seconds delta | "
        "cost delta | tokens delta | iterations delta |".format(
            baseline=baseline_name,
            candidate=candidate_name,
        )
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for task_id in sorted(shared_task_ids, key=sort_key):
        baseline_result = baseline_by_task[task_id]
        candidate_result = candidate_by_task[task_id]
        print(
            "| {task} | {baseline_success} | {candidate_success} | "
            "{seconds_delta:+.2f} | {cost_delta} | {tokens_delta:+d} | "
            "{iterations_delta:+d} |".format(
                task=task_id,
                baseline_success=baseline_result.get("success", False),
                candidate_success=candidate_result.get("success", False),
                seconds_delta=(
                    float(candidate_result.get("duration_seconds", 0.0))
                    - float(baseline_result.get("duration_seconds", 0.0))
                ),
                cost_delta=_format_cost_delta(
                    float(candidate_result.get("cost_usd", 0.0))
                    - float(baseline_result.get("cost_usd", 0.0))
                ),
                tokens_delta=_result_tokens(candidate_result)
                - _result_tokens(baseline_result),
                iterations_delta=(
                    int(candidate_result.get("iterations", 0))
                    - int(baseline_result.get("iterations", 0))
                ),
            )
        )
    if not shared_task_ids:
        print("| No shared tasks |  |  |  |  |  |  |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files.")
    parser.add_argument("baseline", nargs="?", type=Path, help="Baseline results JSON.")
    parser.add_argument("candidate", nargs="?", type=Path, help="Candidate results JSON.")
    parser.add_argument("--baseline-strategy", help="Load latest results for this strategy.")
    parser.add_argument("--candidate-strategy", help="Load latest results for this strategy.")
    return parser.parse_args()


def main_from_paths(baseline_path: Path, candidate_path: Path) -> None:
    baseline = _load_results(baseline_path)
    candidate = _load_results(candidate_path)
    baseline_name = _strategy_name(baseline, baseline_path.stem)
    candidate_name = _strategy_name(candidate, candidate_path.stem)

    print(f"Baseline: {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print()
    _print_summary(baseline, candidate, baseline_name, candidate_name)
    _print_task_deltas(baseline, candidate, baseline_name, candidate_name)


def main() -> None:
    args = parse_args()

    baseline_path = (
        _latest_for_strategy(args.baseline_strategy)
        if args.baseline_strategy
        else args.baseline
    )
    candidate_path = (
        _latest_for_strategy(args.candidate_strategy)
        if args.candidate_strategy
        else args.candidate
    )

    if baseline_path is None or candidate_path is None:
        raise SystemExit(
            "Provide two result files, or use --baseline-strategy and --candidate-strategy."
        )

    main_from_paths(baseline_path, candidate_path)


if __name__ == "__main__":
    main()
