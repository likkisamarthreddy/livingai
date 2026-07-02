# Examples

Runnable, self-contained demos of the Living AI runtime. None require an LLM,
network access, or any third-party package — they simulate agent behavior so you
can see the runtime's mechanics directly.

Run any of them from the `livingai_runtime/` directory:

```bash
python examples/01_basic_checkpoint.py
```

| # | File | Shows |
| --- | --- | --- |
| 01 | [01_basic_checkpoint.py](01_basic_checkpoint.py) | Save a node, attach a compressed state checkpoint, load it back, read write-latency metrics. |
| 02 | [02_crash_recovery.py](02_crash_recovery.py) | Simulate a hard crash (dropped writes), then recover from the last durable checkpoint — replaying idempotent work and **never re-charging the card**. |
| 03 | [03_mock_tools_debugging.py](03_mock_tools_debugging.py) | Replay reasoning with `MOCK_TOOLS`: tool responses come from history, so **zero real API calls** happen during debugging. |
| 04 | [04_cost_tracking.py](04_cost_tracking.py) | Aggregate true token cost from the append-only log, including tokens wasted on failed retries. |
| 05 | [05_langgraph_agent.py](05_langgraph_agent.py) | Instrument a LangGraph-style agent via `LangGraphAdapter`, then recover it — tool nodes are auto-classified as side-effecting. |

## What to look for

- **Example 02** prints `side-effect nodes skipped : 1 (card NOT re-charged)` —
  the core safety guarantee: recovery never re-runs non-idempotent tool calls.
- **Example 03** prints `real API calls during replay: 0` — MOCK_TOOLS lets you
  iterate on prompt logic against frozen tool outputs.
- **Example 05** needs no `langgraph` install; the adapter consumes events as
  plain data. Replace the loop body with real LangGraph callbacks in production.

See the [docs](../docs/README.md) for the concepts behind each demo.
