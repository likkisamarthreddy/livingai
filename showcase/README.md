# Showcase — Drop-in LangGraph integration

LangGraph ships with `SqliteSaver`: a **blocking, single-writer** checkpointer.
Under concurrency it serializes every write behind a file lock, and a slow write
stalls your whole graph.

`livingai` gives you a **non-blocking, horizontally-scalable** replacement in one
line — with a dual-tier hot cache (microsecond reads), zlib compression, and a
50 ms SLA budget so a slow write is *dropped and logged*, never allowed to block
your agent.

## The one-line swap

```diff
- from langgraph.checkpoint.sqlite import SqliteSaver
- saver = SqliteSaver.from_conn_string("agent.db")
+ from showcase.langgraph_redis_saver import LivingAIRedisSaver
+ saver = LivingAIRedisSaver.from_url("redis://localhost:6379")

  graph = builder.compile(checkpointer=saver)
```

Everything else in your LangGraph code stays the same.

## Why swap?

| | LangGraph `SqliteSaver` | `LivingAIRedisSaver` |
| --- | --- | --- |
| Writer model | Single-writer, file lock | Horizontally scalable (Redis) |
| Slow write | Blocks the graph | Dropped after 50 ms, agent continues |
| Recovery reads | Disk query | Hot cache (µs) → Redis fallback |
| State size | Raw | zlib-compressed (60–99% smaller) |
| Concurrent SLA compliance* | ~0.4% | 100% |

\* Measured in [`benchmarks/prod_test_livingai.py`](../benchmarks/README.md) —
50 agents × 20 steps, 50 ms budget.

## Files

- [`langgraph_redis_saver.py`](langgraph_redis_saver.py) — the drop-in
  `BaseCheckpointSaver` implementation (requires `langgraph` installed).
- [`demo_langgraph_redis.py`](demo_langgraph_redis.py) — a **runnable** end-to-end
  demo that needs no LangGraph or Redis server (uses `fakeredis` + the
  `LangGraphAdapter`). Shows crash recovery skipping a non-idempotent tool call.

## Run the demo

```bash
pip install "livingai[redis]" fakeredis
python showcase/demo_langgraph_redis.py
```

```
Running LangGraph agent through the Redis-backed saver:
  plan               [PROMPT idempotent ] -> 'route to the payment tool'
  charge_card_tool   [TOOL   SIDE-EFFECT] -> {'receipt': 'R-1'}
  summarize          [PROMPT idempotent ] -> 'Payment complete.'

💥 Process crashes. Recovering from Redis in a fresh engine:
  resume from    : 'plan'
  replay (safe)  : ['summarize']
  skip (effects) : ['charge_card_tool']

  ✅ The card charge is never replayed — no double billing.
```
