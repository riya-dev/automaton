from __future__ import annotations

import json

import compare_runs
from benchmark_utils import strategy_slug


def _payload(strategy: str, result: dict) -> dict:
    results = [result]
    passed = sum(1 for item in results if item["success"])
    return {
        "created_at": "2026-05-22T00:00:00+00:00",
        "strategy": strategy,
        "summary": {
            "created_at": "2026-05-22T00:00:00+00:00",
            "strategy": strategy,
            "total_tasks": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "average_duration_seconds": result["duration_seconds"],
            "average_iterations": result["iterations"],
            "total_input_tokens": result.get("input_tokens", 0),
            "total_output_tokens": result.get("output_tokens", 0),
            "total_thinking_tokens": result.get("thinking_tokens", 0),
        },
        "results": results,
    }


def test_strategy_slug_matches_filename_expectation() -> None:
    assert strategy_slug("ReAct Tools") == "react-tools"


def test_task_deltas_include_unchanged_success(capsys) -> None:
    baseline = _payload(
        "structured",
        {
            "task_id": "task_001",
            "success": True,
            "duration_seconds": 1.0,
            "cost_usd": 0.10,
            "iterations": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "thinking_tokens": 0,
        },
    )
    candidate = _payload(
        "react",
        {
            "task_id": "task_001",
            "success": True,
            "duration_seconds": 2.5,
            "cost_usd": 0.25,
            "iterations": 3,
            "input_tokens": 30,
            "output_tokens": 15,
            "thinking_tokens": 5,
        },
    )

    compare_runs._print_task_deltas(baseline, candidate, "structured", "react")

    output = capsys.readouterr().out
    assert "| task_001 | True | True | +1.50 | +$0.15000000 | +35 | +2 |" in output


def test_old_result_payload_without_cost_summary_compares(tmp_path, capsys) -> None:
    result = {
        "task_id": "task_001",
        "success": True,
        "duration_seconds": 1.0,
        "cost_usd": 0.01,
        "iterations": 1,
    }
    payload = _payload("structured", result)
    payload["summary"].pop("total_input_tokens")
    payload["summary"].pop("total_output_tokens")
    payload["summary"].pop("total_thinking_tokens")

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    compare_runs.main_from_paths(baseline_path, candidate_path)

    output = capsys.readouterr().out
    assert "| Total cost | $0.01000000 | $0.01000000 | +$0.00000000 |" in output
