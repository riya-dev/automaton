"""Entry point for the Automaton graph."""

from langgraph.graph import END, START, StateGraph

from nodes import planner, coder, executor, critic
from state import AgentState


def critic_router(state: AgentState) -> str:
    """Route to another coding pass unless the run is complete."""
    if state["status"] in {"passed", "failed", "max_iter_reached"}:
        return "end"

    return "coder"


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
        "coder": "coder",
        "end": END,
    },
)

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "task": "Create a simple Flask hello world app",
            "next": None,
            "iteration": 0,
            "max_iterations": 1,
            "messages": [],
            "working_dir": ".",
            "file_tree": "",
            "code_context": "",
            "plan": None,
            "last_edit": None,
            "test_result": None,
            "last_error": None,
            "status": "running",
        }
    )
    print("Graph completed!")
    print(result)
