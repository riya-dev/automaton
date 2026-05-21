"""Entry point for the Automaton graph."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, START, StateGraph

from nodes import planner, coder, executor, critic, evaluator
from state import AgentState


def critic_router(state: AgentState) -> str:
    """Route to another coding pass unless the run is complete."""
    critique = state.get("critique")

    if state["status"] == "passed":
        return "evaluator"

    if critique is None:
        return "evaluator"

    if state["iteration"] >= state["max_iterations"]:
        return "evaluator"

    if critique.verdict == "done":
        return "evaluator"

    if critique.verdict == "give_up":
        return "evaluator"

    if critique.verdict == "replan":
        return "planner"

    if critique.verdict == "continue":
        return "coder"

    return "evaluator"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner)
graph_builder.add_node("coder", coder)
graph_builder.add_node("executor", executor)
graph_builder.add_node("critic", critic)
graph_builder.add_node("evaluator", evaluator)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "coder")
graph_builder.add_edge("coder", "executor")
graph_builder.add_edge("executor", "critic")
graph_builder.add_conditional_edges(
    "critic",
    critic_router,
    {
        "planner": "planner",
        "coder": "coder",
        "evaluator": "evaluator",
    },
)
graph_builder.add_edge("evaluator", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    from langchain_core.tracers.langchain import wait_for_all_tracers

    benchmark_dir = Path("benchmarks/tasks/task_001")
    try:
        result = graph.invoke(
            {
                "task": "Fix next_number so it returns the integer after the input value.",
                "next": None,
                "iteration": 0,
                "max_iterations": 2,
                "messages": [],
                "working_dir": str(benchmark_dir),
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
        )
        print("Graph completed!")
        print(result)
    finally:
        wait_for_all_tracers()
