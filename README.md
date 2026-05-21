# automaton
Autonomous swe agents

## Run
Run the graph:

```bash
python agent.py
```

Run the benchmark harness:

```bash
python eval_harness.py
```

Run a local smoke check without calling Gemini:

```bash
python -m py_compile state.py nodes.py agent.py eval_harness.py
```

## Tracing

To send LangChain/LangGraph traces to LangSmith, set:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=Automaton
```

## Benchmark Results

Latest local run:

Summary:

- total tasks: 10
- passed: 10
- failed: 0
- pass rate: 100.0%
- average iterations: 1.0
- average duration: 18.57s

| task | category | success | status | verdict | iterations | seconds |
| --- | --- | --- | --- | --- | ---: | ---: |
| task_001 | off-by-one | True | passed | done | 1 | 17.33 |
| task_002 | off-by-one | True | passed | done | 1 | 14.34 |
| task_003 | missing import | True | passed | done | 1 | 4.96 |
| task_004 | wrong return type | True | passed | done | 1 | 17.86 |
| task_005 | logic error | True | passed | done | 1 | 50.53 |
| task_006 | missing function | True | passed | done | 1 | 44.32 |
| task_007 | bad default argument | True | passed | done | 1 | 10.24 |
| task_008 | string normalization bug | True | passed | done | 1 | 8.74 |
| task_009 | date edge case | True | passed | done | 1 | 10.69 |
| task_010 | exception handling bug | True | passed | done | 1 | 6.71 |
