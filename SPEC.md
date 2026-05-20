# Automaton: Autonomous Coding Agent — Spec & Roadmap

## What It Does

Given a high-level task or a repo with failing tests, Automaton:

1. Scans the codebase to build context
2. Plans a concrete list of subtasks
3. Writes or edits code
4. Runs tests in a sandbox
5. Critiques the result and either fixes it or escalates
6. Scores the final solution with an LLM judge
7. Returns a full trajectory report with cost + latency metrics

---

## Architecture

### Directory Structure

```
automaton/
├── src/
│   └── automaton/
│       ├── __init__.py
│       ├── graph.py              # Main LangGraph graph
│       ├── state.py              # AgentState + all Pydantic models
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── context_builder.py  # Pre-planner: scans repo, builds file tree
│       │   ├── planner.py
│       │   ├── coder.py
│       │   ├── executor.py
│       │   ├── critic.py
│       │   └── evaluator.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── filesystem.py     # read_file, write_file, list_dir, apply_diff, search_code
│       │   └── execution.py      # run_command with safety layer
│       ├── models/
│       │   ├── __init__.py
│       │   └── gemini.py         # Model tiering: Flash vs Pro routing
│       └── observability/
│           ├── __init__.py
│           └── tracking.py       # Cost + latency per node
├── benchmarks/
│   ├── tasks/                    # 10–20 broken repos with failing tests
│   └── run_benchmark.py
├── demo/
│   └── app.py                    # Streamlit UI
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

The existing `graph.py` and `tools.py` will be refactored into this structure on Day 1.

---

### State Design

```python
class AgentState(TypedDict):
    # Core
    task: str                          # Original user task
    messages: Annotated[List, operator.add]

    # Context
    file_tree: str                     # Snapshot of repo structure
    code_context: dict[str, str]       # {filepath: content} for relevant files

    # Plan
    plan: Plan | None                  # Structured plan (see models below)
    current_step_index: int

    # Execution
    last_edit: CodeEdit | None         # What the coder last changed
    test_result: TestResult | None     # Structured test output (pass/fail per test)
    last_error: str | None

    # Critique
    critique: CritiqueResult | None    # Critic's verdict + confidence score

    # Control
    iteration: int
    max_iterations: int                # Default 8
    status: Literal["running", "passed", "failed", "max_iter_reached"]

    # Observability
    trajectory: list[TrajectoryStep]   # Full action history for the report
    total_cost_usd: float
    total_latency_ms: float
```

### Pydantic Models (Instructor + structured outputs)

```python
class PlanStep(BaseModel):
    step_id: int
    action: Literal["read", "write", "test", "analyze", "search"]
    target_file: str | None
    description: str

class Plan(BaseModel):
    goal_summary: str
    steps: list[PlanStep]

class CodeEdit(BaseModel):
    file_path: str
    edit_type: Literal["create", "modify", "delete"]
    content: str                       # Full content for create; unified diff for modify

class TestResult(BaseModel):
    passed: bool
    total_tests: int
    passed_count: int
    failed_count: int
    failures: list[TestFailure]        # {test_name, error_message, traceback}
    raw_output: str

class CritiqueResult(BaseModel):
    summary: str
    issues: list[str]
    confidence: float                  # 0.0–1.0
    verdict: Literal["continue", "replan", "done", "give_up"]

class EvalResult(BaseModel):
    correctness: float                 # 0–10
    code_quality: float                # 0–10
    efficiency: float                  # 0–10
    trajectory_efficiency: float       # iterations used / max_iterations (lower is better)
    notes: str
```

---

### Graph (Nodes + Edges)

```
START
  └─► context_builder          # Scans repo, populates file_tree + code_context
        └─► planner            # Gemini Pro → structured Plan
              └─► coder        # Gemini Flash → CodeEdit (diff or full file)
                    └─► executor  # Run tests, parse output → TestResult
                          └─► critic  # Gemini Pro → CritiqueResult
                                └─► [conditional routing]
                                      ├─ "continue"  → coder
                                      ├─ "replan"    → planner  (critic says plan is wrong)
                                      ├─ "done"      → evaluator
                                      └─ "give_up"   → evaluator
                                            └─► END
```

**Routing logic in `critic_router`:**
- `verdict == "done"` or `test_result.passed` → evaluator
- `verdict == "continue"` and `iteration < max_iterations` → coder
- `verdict == "replan"` → planner (passes critique + failure context)
- `verdict == "give_up"` or `iteration >= max_iterations` → evaluator (with failure status)

The key addition over the original spec is the **"replan" edge** back to Planner. Sometimes the fix isn't "write better code" — the plan itself is wrong. The Critic can signal this explicitly.

---

### Model Tiering

| Node           | Model         | Reason                                      |
|----------------|---------------|---------------------------------------------|
| context_builder | Flash        | Fast file scanning, no deep reasoning needed |
| planner        | Pro           | Complex task decomposition                  |
| coder          | Flash         | Fast iteration; most expensive in loop count|
| executor       | (no LLM)      | Pure subprocess + output parsing            |
| critic         | Pro           | Nuanced judgment, confidence scoring        |
| evaluator      | Pro           | Final LLM-as-judge scoring                  |

---

### Tools

**Filesystem (`tools/filesystem.py`):**
- `read_file(path)` — exists, keep
- `write_file(path, content)` — exists, add path validation (no `..` traversal)
- `list_directory(path, depth=2)` — returns annotated file tree
- `apply_diff(path, unified_diff)` — applies a unified diff rather than full rewrite; more token-efficient for edits
- `search_code(pattern, directory)` — grep-style symbol/string search

**Execution (`tools/execution.py`):**
- `run_command(command, working_dir, timeout=60)` — the current version is unsafe
- Add allowlist: `pytest`, `python`, `npm test`, `cargo test`, `go test`, `make test`
- Add blocklist check: `rm -rf`, `sudo`, `curl`, `wget`, `ssh`
- Lock working directory: commands can't escape the sandbox path
- Parse output into `TestResult` (detect pytest, jest, cargo test formats)

---

### Observability

- **LangSmith**: wrap every node with `@traceable`; set `LANGCHAIN_TRACING_V2=true`
- **Cost tracking**: after each LLM call, extract `usage_metadata` and accumulate into `state.total_cost_usd`
- **Trajectory**: each node appends a `TrajectoryStep(node, action_summary, tokens, latency_ms)` to `state.trajectory`
- **Final report**: Evaluator node generates a JSON report with trajectory + eval scores

---

## Day-by-Day Roadmap

### Day 1 — Structure, State, and Skeleton

**Goal**: A runnable graph with proper structure, models defined, and real Gemini calls in at least one node.

Tasks:
1. Refactor `graph.py` and `tools.py` into the `src/automaton/` package structure
2. Write `pyproject.toml` with all deps (`langgraph`, `langchain-google-genai`, `instructor`, `pydantic>=2`, `langsmith`, `docker`, `streamlit`)
3. Define all Pydantic models in `state.py` (Plan, CodeEdit, TestResult, CritiqueResult, EvalResult, AgentState)
4. Implement `context_builder` node (real: `list_directory` + selective `read_file`)
5. Implement `planner` node with Gemini Pro + Instructor structured output → `Plan`
6. Keep `coder`, `executor`, `critic` as stubs that print + pass state through
7. Wire the full graph with all conditional edges (including the replan edge)
8. Verify the graph compiles and the planner produces a real `Plan` object

Deliverables: `src/automaton/` package, `state.py`, `graph.py` with real planner, `pyproject.toml`

---

### Day 2 — Safe Execution + Coder Node

**Goal**: The agent can actually write code and run tests safely.

Tasks:
1. Implement `tools/execution.py` with the safety layer (allowlist, blocklist, working dir lock, timeout)
2. Implement `tools/filesystem.py` with `list_directory`, `apply_diff`, `search_code` (+ harden existing tools)
3. Implement the `coder` node with Gemini Flash + Instructor → `CodeEdit`
   - For new files: emit full content
   - For edits: emit unified diff (apply with `apply_diff` tool)
4. Implement the `executor` node: call `run_command`, parse output into `TestResult`
   - Support pytest output format first; jest + cargo as stretch
5. Create `benchmarks/tasks/task_001/` — a simple broken Python repo with 3 failing tests
6. Run the graph end-to-end (planner → coder → executor) and verify it writes a file + runs tests

Deliverables: safe `run_command`, full tool suite, working coder + executor nodes, first benchmark task

---

### Day 3 — Critic + Self-Critique Loop

**Goal**: The agent can critique its own output and decide whether to loop, replan, or stop.

Tasks:
1. Implement `critic` node with Gemini Pro + Instructor → `CritiqueResult`
   - Input: test_result, last_edit, code_context, iteration count
   - Output: summary, issues list, confidence (0–1), verdict (continue/replan/done/give_up)
2. Implement `critic_router` conditional edge with all four paths
3. Add the `replan` path: Critic can send back to Planner with failure context injected into state
4. Implement `evaluator` node: final LLM-as-judge scoring → `EvalResult`
5. Add 5 more benchmark tasks (varying difficulty: fix a bug, add a feature, refactor)
6. Run 3 tasks end-to-end with the full loop; manually verify the routing works

Deliverables: critic node, all routing, evaluator node, 6 benchmark tasks total

---

### Day 4 — Observability + Benchmark Harness

**Goal**: Full cost/latency visibility and a benchmark suite that can run automatically.

Tasks:
1. Wire LangSmith tracing (`LANGCHAIN_TRACING_V2`, `@traceable` on all nodes)
2. Implement `observability/tracking.py`: per-node token counts, latency, cost accumulation into state
3. Implement `benchmarks/run_benchmark.py`:
   - Loops over all tasks in `benchmarks/tasks/`
   - Runs the graph on each
   - Collects: success (bool), iterations used, total cost, total latency, eval scores
   - Outputs a JSON results file + prints a markdown table
4. Add 10+ more benchmark tasks to reach 15–20 total
5. Run the full benchmark suite; capture baseline metrics

Deliverables: LangSmith integration, cost tracking, benchmark runner, 15–20 tasks, baseline metrics table

---

### Day 5 — UI, Polish, and README

**Goal**: A demo-ready agent with a clean UI and excellent documentation.

Tasks:
1. Build `demo/app.py` with Streamlit:
   - Left panel: task input + run button
   - Center: live node execution feed (streaming graph events)
   - Right panel: trajectory viewer + final eval scores
   - Bottom: cost + latency breakdown per node
2. Add trajectory report generation: after evaluator, write a JSON + Markdown report to disk
3. Write the full README:
   - Architecture diagram (Mermaid)
   - Setup instructions
   - Metrics table from benchmark run
   - Example trajectory (input → plan → edits → test result → eval score)
4. Clean up: remove debug prints, add `.env.example`, ensure `pyproject.toml` has all deps pinned

Deliverables: Streamlit demo, trajectory reports, complete README with metrics

---

## Deviations from Original Plan (and Why)

| Original | Change | Reason |
|---|---|---|
| Coder always loops back | Add "replan" edge from Critic → Planner | Sometimes the plan is wrong, not the code; routing back to coder is wasted iterations |
| run_command as-is | Add allowlist + blocklist + working dir lock | Current version can execute arbitrary shell; a safety layer is required even for local dev |
| Full file rewrites | Unified diffs for modifications | Diffs are 3–5x cheaper in tokens and less destructive; full rewrites on large files lose context |
| context_builder implicit | Explicit node before Planner | Planner needs file tree + relevant file contents before it can plan; separating this makes it testable |
| LangSmith only | LangSmith + per-state cost accumulation | LangSmith traces are great for debugging but don't give you a final cost summary in the state; need both |
| Docker on Day 2 | Docker as stretch goal, subprocess with safety first | Docker adds significant setup complexity; a hardened subprocess layer covers 90% of the safety need for Day 1–2 |

---

## Success Criteria

| Metric | Target |
|---|---|
| Benchmark success rate | ≥ 70% of tasks passing all tests |
| Average iterations to success | ≤ 4 |
| p50 latency per task | ≤ 45s |
| p95 latency per task | ≤ 120s |
| Average cost per task | ≤ $0.05 |
| Trajectory efficiency | ≥ 0.6 (iterations used / max) |

---

## Dependencies

```toml
[project]
dependencies = [
    "langgraph>=0.2",
    "langchain-google-genai>=2.0",
    "langchain-core>=0.3",
    "instructor>=1.4",
    "pydantic>=2.7",
    "langsmith>=0.1",
    "python-dotenv>=1.0",
    "docker>=7.0",          # stretch: Day 2+
    "streamlit>=1.35",
    "pytest>=8.0",          # for benchmark tasks
    "rich>=13.0",           # CLI output formatting
]
```

---

## Open Questions / Decisions Deferred

- **Docker vs subprocess**: Start with subprocess + safety layer. Add Docker containerization in a follow-up if benchmark tasks need it (e.g., tasks that install packages).
- **Long-term memory**: MemorySaver covers the session. A vector store (ChromaDB/Qdrant) for cross-session memory is a stretch goal after Day 5.
- **Multi-language support**: Benchmark tasks will be Python-only initially. Jest/cargo test parsing can be added incrementally.
- **Parallelism**: Planner could dispatch independent subtasks to parallel Coder instances. Deferred — sequential is simpler and easier to debug first.
