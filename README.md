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

## Benchmark Results

Latest local run:

| task | category | success | status | verdict | iterations | seconds |
| --- | --- | --- | --- | --- | ---: | ---: |
| task_001 | off-by-one | True | passed | done | 1 | 4.7 |
| task_002 | off-by-one | True | passed | done | 1 | 5.44 |
| task_003 | missing import | True | passed | done | 1 | 11.89 |
| task_004 | wrong return type | True | passed | done | 1 | 12.82 |
| task_005 | logic error | True | passed | done | 1 | 10.44 |
| task_006 | missing function | True | passed | done | 1 | 5.16 |
| task_007 | bad default argument | True | passed | done | 1 | 6.07 |
| task_008 | string normalization bug | True | passed | done | 1 | 3.99 |
| task_009 | date edge case | True | passed | done | 1 | 3.47 |
| task_010 | exception handling bug | True | passed | done | 1 | 3.05 |
