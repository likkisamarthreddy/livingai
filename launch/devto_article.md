---
title: "I built a crash recovery runtime for AI agents — zero dependencies, works with LangGraph/CrewAI/OpenAI"
published: true
description: "livingai records every agent step to an append-only log so any run can be recovered after a crash without re-running side effects."
tags: python, ai, agents, opensource
---

# The Problem

AI agents fail in expensive ways.

A process crashes after the LLM reasoned, after the tool charged a card, three steps into a ten-step plan. Everything is gone. On restart you repay for the tokens and hope the tool doesn't fire its side effects twice.

The existing solutions are framework-specific, require a managed database, or don't address the re-execution problem at all.

# What I Built

**[livingai](https://github.com/likkisamarthreddy/livingai)** — a crash recovery, checkpointing, and replay runtime for AI agents.

Zero runtime dependencies. Pure Python standard library. Works with LangGraph, CrewAI, and the OpenAI Agents SDK.

```bash
pip install livingai
```

## How Recovery Works

Every agent step is recorded to an append-only SQLite log. On crash, the recovery engine reads the log and builds a `RecoveryPlan`:

```python
from livingai import CheckpointEngine, RecoveryEngine, SQLiteStore

engine = CheckpointEngine(SQLiteStore("agent.db"))
recovery = RecoveryEngine(engine)

plan = await recovery.plan("run-1")
print(f"Resume from: {plan.resume_node_id}")
print(f"Safe to replay: {len(plan.replay_nodes)} nodes")
print(f"Skipping side effects: {len(plan.skipped_nodes)} nodes")
```

The key insight: `TOOL` nodes default to **non-idempotent**. The recovery engine knows never to re-run a card charge, email send, or API write. Only `PROMPT` and `MEMORY` nodes (idempotent by nature) are replayed.

## Replay Modes

Beyond crash recovery, the library has four replay modes:

- **FULL** — replay the entire execution
- **FROM_NODE** — resume from a specific checkpoint
- **MOCK_TOOLS** — return stored tool responses without real API calls (the debugging superpower)
- **COUNTERFACTUAL** — re-run with different inputs to understand what would have changed

`MOCK_TOOLS` is the one I use constantly. Instead of paying for 20 API calls while debugging reasoning, you replay with stored responses. Zero cost, instant feedback.

## Framework Adapters

The core has no framework imports. Adapters are thin wrappers that tag nodes with framework metadata:

```python
from livingai.adapters import LangGraphAdapter, CrewAIAdapter, OpenAIAgentsAdapter
```

The same recovery guarantees work regardless of which framework you use.

## The Architecture

```
ExecutionNode ──► CheckpointStore (Tier 2: durable, append-only SQLite)
      ▲                  ▲
      │                  │
 Adapters          CheckpointEngine ──► HotCache (Tier 1: LRU + TTL, ~4µs reads)
 (LangGraph/             │
  CrewAI/           RecoveryEngine ──► RecoveryPlan (replay vs. skip)
  OpenAI)           ReplaySession  ──► FULL / FROM_NODE / MOCK_TOOLS / COUNTERFACTUAL
```

## Performance

Checkpointing is on the hot path of every agent step.

| | |
|---|---|
| Write p50 | ~0.3ms |
| Write p99 | ~1ms |
| Hot read | ~4µs |
| Compression | 60–99% |

The overhead budget is enforced: a write that would exceed 50ms is dropped rather than blocking the agent.

## Optional Backends

The default is SQLite (zero config). Swap it for Redis or PostgreSQL:

```python
from livingai.stores.redis import RedisStore
from livingai.stores.postgres import PostgresStore

# Redis
engine = CheckpointEngine(RedisStore(url="redis://localhost:6379"))

# PostgreSQL
store = PostgresStore(dsn="postgresql://user:pass@localhost/db")
await store.initialize()
engine = CheckpointEngine(store)
```

## Quality

- 128 tests, 100% line coverage (including crash simulation and 10k-node stress tests)
- `mypy --strict` clean across all source files
- CI on Python 3.9, 3.10, 3.11, 3.12
- Apache-2.0 license

## What's Next

FastAPI cloud backend (5 endpoints), cloud sync client, and a web replay dashboard. The storage protocol is already defined — Redis and PostgreSQL backends are already in.

---

If you're building agents that need to survive crashes, or if you want to debug agent reasoning without burning API credits, give it a try:

```bash
pip install livingai
```

GitHub: **https://github.com/likkisamarthreddy/livingai**

I'd love feedback on the API design, the recovery semantics, or anything that would make this more useful for your workflow.
