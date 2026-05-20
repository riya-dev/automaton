from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import operator
from tools import read_file, write_file, run_command

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next: str
    code_context: str
    last_error: str | None
    plan: str | None
    critique: str | None
    iteration: int
    max_iterations: int = 8

tools = [read_file, write_file, run_command]

def planner(state: AgentState):
    print("→ Planner node")
    return {"next": "coder"}

def coder(state: AgentState):
    print("→ Coder node")
    return {"next": "executor"}

def executor(state: AgentState):
    print("→ Executor node")
    return {"next": "critic"}

def critic(state: AgentState):
    print("→ Critic node")
    state["iteration"] = state.get("iteration", 0) + 1
    if state["iteration"] >= state["max_iterations"]:
        return {"next": "end"}
    return {"next": "coder"}  # loop back

# Build graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner)
graph_builder.add_node("coder", coder)
graph_builder.add_node("executor", executor)
graph_builder.add_node("critic", critic)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "coder")
graph_builder.add_edge("coder", "executor")
graph_builder.add_edge("executor", "critic")

# Conditional routing from critic
def critic_router(state: AgentState):
    if state.get("iteration", 0) >= state["max_iterations"]:
        return END
    return "coder"

graph_builder.add_conditional_edges("critic", critic_router)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Quick test
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage(content="Create a simple Flask hello world app")],
        "iteration": 0,
        "code_context": "",
        "last_error": None
    }, config={"configurable": {"thread_id": "test-1"}})
    print("✅ Graph completed!")
