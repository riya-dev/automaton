"""Entry point for the Automaton graph."""

from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

from nodes import planner, coder, executor, critic
from state import AgentState


def critic_router(state: AgentState) -> str:
    """Route to another coding pass unless the run is complete."""
    critique = state.get("critique")

    if state["status"] == "passed":
        return "end"

    if critique is None:
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        return "end"

    if critique.verdict == "done":
        return "end"

    if critique.verdict == "give_up":
        return "end"

    if critique.verdict == "replan":
        return "planner"

    if critique.verdict == "continue":
        return "coder"

    return "end"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner)
graph_builder.add_node("coder", coder)
graph_builder.add_node("executor", executor)
graph_builder.add_node("critic", critic)

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
        "end": END,
    },
)

graph = graph_builder.compile()


if __name__ == "__main__":
    benchmark_dir = Path("benchmarks/tasks/task_001")
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
        }
    )
    print("Graph completed!")
    print(result)
