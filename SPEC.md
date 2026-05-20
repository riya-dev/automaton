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

**MVP goal**: fix a real failing pytest repo end-to-end before adding any polish.

---

## Directory Structure

```
automaton/
├── agent.py              # Graph definition + entry point (run from here)
├── state.py              # AgentState + Pydantic models as they are introduced
├── nodes.py              # Node functions (planner, coder, executor, critic; evaluator on Day 4)
├── tools.py              # read_file, write_file, list_dir, run_command
├── eval_harness.py       # Benchmark runner + LLM-as-judge scoring
├── benchmarks/
│   └── tasks/            # Each task: broken repo with failing pytest suite
├── demo/
│   └── app.py            # Streamlit UI (Day 5)
├── pyproject.toml
├── .env.example
└── README.md
```

Start flat. Refactor into subpackages only if a single file becomes unmanageable.

---

## State Design

Start with the smallest state that can prove the graph runs. Add fields only when a node actually needs them.

**Day 1 minimum:**

```python
class AgentState(TypedDict):
    task: str
    next: str | None
    iteration: int
    max_iterations: int
```

**Day 2 loop state:**

```python
class AgentState(TypedDict):
    # Core
    task: str
    messages: Annotated[List, operator.add]
    working_dir: str                   # Locked sandbox path

    # Context (populated by Planner before generating a plan)
    file_tree: str                     # Output of list_dir
    code_context: str                  # Concatenated content of relevant files (flat string for MVP)

    # Plan
    plan: Plan | None

    # Execution
    last_edit: CodeEdit | None
    test_result: TestResult | None
    last_error: str | None

    # Critique
    critique: CritiqueResult | None

    # Control
    iteration: int
    max_iterations: int                # Default 6
    status: Literal["running", "passed", "failed", "max_iter_reached"]

    # Add on Day 4:
    # trajectory: list[TrajectoryStep]
    # eval_result: EvalResult | None
    # total_cost_usd: float
    # total_latency_ms: float
```

## Pydantic Models

Keep these lean for MVP. On Day 1, no Pydantic models are required; stubs can return plain strings or fixed routing values.

**Day 2 models:**

```python
class PlanStep(BaseModel):
    step_id: int
    description: str
    target_file: str | None

class Plan(BaseModel):
    goal_summary: str
    steps: list[PlanStep]

class CodeEdit(BaseModel):
    file_path: str
    content: str                       # Full file content (no diffs for MVP)

class TestResult(BaseModel):
    passed: bool
    passed_count: int
    failed_count: int
    failure_summary: str               # Raw truncated pytest output
    raw_output: str
```

**Day 3 model:**

```python

class CritiqueResult(BaseModel):
    summary: str
    issues: list[str]
    confidence: float                  # 0.0–1.0
    verdict: Literal["continue", "replan", "done", "give_up"]
```

**Day 4 models:**

```python
class TrajectoryStep(BaseModel):
    node: str
    summary: str
    latency_ms: float | None = None
    tokens: int | None = None

class EvalResult(BaseModel):
    correctness: float                 # 0–10
    code_quality: float                # 0–10
    trajectory_efficiency: float       # iterations used / max_iterations
    notes: str
```

`apply_diff` is a post-MVP optimization. Full rewrites are fine for the benchmark task sizes.

---

## Graph

Target graph by Day 4. On Day 1, use a simple linear stub graph; on Day 3, `done` / `give_up` can route directly to `END` until the evaluator is added.

```
START
  └─► planner        # Gemini Pro → Plan (reads file_tree + code_context)
        └─► coder    # Gemini Flash → CodeEdit
              └─► executor   # run_command pytest → TestResult
                    └─► critic   # Gemini Pro → CritiqueResult
                          └─► [critic_router]
                                ├─ "continue"  → coder
                                ├─ "replan"    → planner   ← key edge
                                ├─ "done"      → evaluator
                                └─ "give_up"   → evaluator
                                      └─► END
```

**`critic_router` logic:**
- `test_result.passed` or `verdict == "done"` → evaluator
- `verdict == "continue"` and `iteration < max_iterations` → coder
- `verdict == "replan"` → planner (critique + error context injected into state)
- `verdict == "give_up"` or `iteration >= max_iterations` → evaluator (failure status)

The **replan edge** is the critical design choice. When the fix requires rethinking the approach — not just rewriting the code — routing back to Planner avoids wasted iterations. Context builder is folded into the Planner node for MVP (it reads the file tree itself before generating a plan).

---

## Model Tiering

| Node      | Model        | Reason                                          |
|-----------|--------------|-------------------------------------------------|
| planner   | Gemini Pro   | Complex task decomposition, needs deep reasoning |
| coder     | Gemini Flash | Fast iteration; runs every loop, cost-sensitive  |
| critic    | Gemini Pro   | Nuanced judgment, confidence scoring             |
| evaluator | Gemini Pro   | Final LLM-as-judge; quality over speed           |
| executor  | (no LLM)     | Pure subprocess + regex parsing                  |

---

## Tools (`tools.py`)

All tools in one file for MVP. Keep them simple and defensive.

- `read_file(path)` — reads a file relative to working_dir
- `write_file(path, content)` — writes full file content; validates no `..` path traversal
- `list_dir(path, depth=2)` — annotated file tree string
- `run_command(command, working_dir, timeout=60)` — subprocess with:
  - Allowlist: `pytest`, `python`, `python3`, `npm test`, `make test`
  - Blocklist: `rm`, `sudo`, `curl`, `wget`, `ssh`, `git push`
  - Working directory locked to sandbox path
  - Returns stdout + stderr as string (Executor parses into TestResult)

No `apply_diff` or `search_code` for MVP. Add after the loop is working end-to-end.

---

## Observability

- **Day 1–3**: rely on simple prints and optional plain-list trajectory notes while the loop is still taking shape
- **Day 4**: introduce `TrajectoryStep`, benchmark results JSON, and the evaluator report
- **LangSmith**: set `LANGCHAIN_TRACING_V2=true`; add `with langsmith.trace(...)` around graph invoke
- **Day 5 stretch**: accumulate `total_cost_usd` in state after each LLM call using `response.usage_metadata`

---

## Day-by-Day Roadmap

### Day 1 — Flat Structure + Stub Graph That Runs

**Goal**: A graph that executes end-to-end with stub nodes. No LLM calls yet. Prove the wiring works.

1. Create the flat files: `agent.py`, `state.py`, `nodes.py`
2. Define the minimal Day 1 `AgentState`: `task`, `next`, `iteration`, `max_iterations`
3. Stub all nodes in `nodes.py` — each prints its name and returns a small state update
4. Wire the graph in `agent.py` with simple linear edges: planner → coder → executor → critic → END
5. Run `python agent.py` and confirm it traverses all nodes without error

**Deliverables**: runnable graph, stub nodes, minimal `AgentState`

---

### Day 2 — Working End-to-End Loop (crude is fine)

**Goal**: Agent makes real LLM calls, writes a file, runs pytest, and loops. Even if it's clunky, the loop closes.

1. Implement real `planner` node: Gemini Pro + Instructor → `Plan`; reads `file_tree` in-node with `list_dir`
2. Implement real `coder` node: Gemini Flash + Instructor → `CodeEdit`; writes file with `write_file`
3. Implement `executor` node: calls `run_command("pytest --tb=short")`, parses exit code + stdout into `TestResult`
4. Define only the Day 2 Pydantic models: `Plan`, `PlanStep`, `CodeEdit`, `TestResult`
5. Implement `tools.py`: `list_dir`, hardened `read_file` / `write_file`, basic `run_command` with allowlist + blocklist
6. Write `pyproject.toml` with the dependencies needed for the loop
7. Create `benchmarks/tasks/task_001/`: **extremely simple** — one Python file with a single off-by-one bug, one failing test
8. Run planner → coder → executor with `max_iterations=2`; verify tests go from failing → passing

The first task should be trivially fixable so you can confirm the loop works before adding complexity.

**Deliverables**: real planner + coder + executor, minimal models, tools with basic safety layer, first benchmark task (1 bug, 1 test), closed loop confirmed

---

### Day 3 — Critic + Replan

**Goal**: The full self-critique loop is live. Agent can replan when stuck.

1. Implement `critic` node: Gemini Pro + Instructor → `CritiqueResult` (summary, issues, confidence, verdict)
2. Implement `critic_router` with all four paths
3. Test the replan edge explicitly: craft a task where the initial plan is wrong and verify it reroutes
4. Add lightweight trajectory logging as plain dicts or strings if useful for debugging; do not build the full `TrajectoryStep` model yet
5. Add 4 more benchmark tasks (vary: off-by-one bug, missing import, wrong return type, logic error)

**Deliverables**: full critique loop, replan edge tested, 5 benchmark tasks total

---

### Day 4 — Benchmark Harness

**Goal**: Automated runner that scores the agent across multiple tasks. Observability is basic (success + iterations).

1. Build `eval_harness.py`:
   - Loops over `benchmarks/tasks/`
   - Runs agent on each; collects: success (bool), iterations used, duration, final status
   - Prints markdown results table + writes `benchmark_results.json`
2. Add `TrajectoryStep` and append one record per node with node name, summary, and optional latency
3. Implement `evaluator` node: LLM-as-judge → `EvalResult`; writes `trajectory_report.json`
4. Add 5–10 more tasks to reach 10–15 total (vary bugs: wrong return value, missing function, logic error, import error)
5. Run full benchmark; capture baseline metrics for README
6. Wire LangSmith tracing (set `LANGCHAIN_TRACING_V2=true`, wrap graph invoke — minimal effort, high observability value)

**Deliverables**: benchmark runner, `TrajectoryStep`, `EvalResult`, evaluator node, 10–15 tasks, baseline metrics table, LangSmith tracing wired

---

### Day 5 — Demo + README + Stretch Observability

**Goal**: Demo-ready. Anyone can clone, run, and understand the results.

1. Write the full README:
   - Mermaid architecture diagram
   - Setup + usage instructions
   - Metrics table from benchmark run
   - Example trajectory (input → plan → edits → test result → eval)
2. Build `demo/app.py` (Streamlit) — keep it simple:
   - Task input + run button
   - Live node execution feed (stream graph events)
   - Final eval score display
3. Add `.env.example`, clean up debug prints, pin deps in `pyproject.toml`

**Stretch (only if Days 1–4 finished cleanly):**
- Add `total_cost_usd: float` and token counts to `AgentState`
- Accumulate cost per LLM call using `response.usage_metadata`
- Show cost + latency breakdown in the Streamlit UI

**Deliverables**: complete README with metrics, Streamlit demo

---

### Day 6 — Polish (if time allows)

- `apply_diff` tool for token-efficient edits on larger files
- `search_code` tool (grep-style) for symbol lookup
- Full trajectory report as JSON + Markdown
- Docker sandbox for full isolation
- Multi-framework test parsing (jest, cargo test)
- Long-term vector store memory (ChromaDB) for cross-session context

---

## Success Criteria

| Metric                   | Target                                    |
|--------------------------|-------------------------------------------|
| Benchmark success rate   | ≥ 70% of tasks passing all tests          |
| Average iterations       | ≤ 4 per successful task                   |
| p50 latency per task     | ≤ 45s                                     |
| p95 latency per task     | ≤ 120s                                    |
| Average cost per task    | ≤ $0.05                                   |
| Trajectory efficiency    | ≥ 0.6 (iterations used / max\_iterations) |

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
    "streamlit>=1.35",
    "pytest>=8.0",
    "rich>=13.0",
]
# Day 6 stretch:
# "docker>=7.0"
```