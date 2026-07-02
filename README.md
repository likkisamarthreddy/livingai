# Living AI

**Crash recovery, checkpointing, and replay for AI agents — one runtime that works across LangGraph, CrewAI, and OpenAI Agents.**

[![CI](https://github.com/likkisamarthreddy/livingai/actions/workflows/ci.yml/badge.svg)](https://github.com/likkisamarthreddy/livingai/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/livingai)](https://pypi.org/project/livingai/)
[![Python versions](https://img.shields.io/pypi/pyversions/livingai)](https://pypi.org/project/livingai/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](#quality)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](#quality)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Runtime deps](https://img.shields.io/badge/runtime%20deps-0-blueviolet)](pyproject.toml)

---

## The problem

AI agents crash. A process dies mid-workflow — after the LLM reasoned, after the
tool charged a card, three steps into a ten-step plan — and all of that work is
gone. You restart from zero, pay for the tokens again, and hope the tool doesn't
fire its side effects twice. And when something goes wrong, you can't replay what
happened to understand *why*.

## The solution

Living AI records every step of an agent execution to an **append-only log**, so
any run can be:

- **Recovered** — resume from the last durable checkpoint after a crash, replaying
  only the *idempotent* work and never re-running side-effecting tool calls
  (payments, emails, API writes).
- **Replayed** — re-run a recorded execution for debugging, with `MOCK_TOOLS` mode
  returning recorded tool responses so you can iterate on reasoning without real
  API calls.
- **Audited** — inspect cost, latency, and the full node graph of any run.

## Why it's different

Most observability and checkpointing tools lock you into one framework. Living AI
ships a **framework-agnostic core** with thin adapters for all three major agent
frameworks — the same recovery guarantees whether you use LangGraph, CrewAI, or
the OpenAI Agents SDK:

```python
from livingai.adapters import LangGraphAdapter, CrewAIAdapter, OpenAIAgentsAdapter
```

And the core has **zero runtime dependencies** — it's pure standard library.

## Install

```bash
pip install livingai
```

## Crash recovery in 18 lines

```python
import asyncio
from livingai import (
    CheckpointEngine, ExecutionNode, NodeType, RecoveryEngine, SQLiteStore, Status,
)


async def main():
    engine = CheckpointEngine(SQLiteStore("agent.db"))

    # Your agent checkpoints after an expensive step.
    step = ExecutionNode(execution_id="run-1", type=NodeType.PROMPT,
                         status=Status.SUCCESS, output="plan ready")
    await engine.save(step, state=b"...serialized agent state...")

    # A tool with real side effects runs (e.g. charging a card).
    charge = ExecutionNode(execution_id="run-1", type=NodeType.TOOL,
                           status=Status.SUCCESS, output={"receipt": "R-1"})
    await engine.save(charge)

    # 💥 The process crashes. On restart, recover from the durable log:
    recovery = RecoveryEngine(CheckpointEngine(SQLiteStore("agent.db")))
    plan = await recovery.plan("run-1")
    print("resume from :", plan.resume_node_id)        # last durable checkpoint
    print("replay safe :", len(plan.replay_nodes))     # idempotent work to redo
    print("skip effects:", len(plan.skipped_nodes))    # card is NOT re-charged


asyncio.run(main())
```

```
resume from : d482c31e-...
replay safe : 0
skip effects: 1          # the card is never charged twice
```

The [`examples/`](examples/README.md) directory has five runnable demos (crash
recovery, `MOCK_TOOLS` debugging, cost tracking, and the LangGraph adapter) — none
require an LLM or network.

## Performance

Checkpointing is on the hot path of every agent step, so it has to be fast. It is.

| Metric | Result | Notes |
| --- | --- | --- |
| Checkpoint write (p50) | **~0.3 ms** | 50 KB compressed state blob |
| Checkpoint write (p95) | **~0.8 ms** | |
| Checkpoint write (p99) | **~1 ms** | ~50× under the 50 ms budget |
| Hot recovery read | **~4 µs** | vs ~190 µs cold — ~40× faster |
| Compression | **60–99%** | typical agent state (histories, docs) |

Measured on a dev laptop with the **default 50 ms overhead budget**, 50 KB blobs,
2000 writes — the same configuration you get out of the box.

Reproduce all of it yourself. The [`benchmarks/`](benchmarks/README.md) directory
ships a three-tier suite you can run in one command each:

```bash
pip install "livingai[redis]" fakeredis

python benchmarks/benchmark_livingai.py        # Big Test — 1 agent, 500 steps, 250 KB
python benchmarks/prod_test_livingai.py        # Production — 50 agents, SQLite vs Redis
python benchmarks/hyperscale_test_livingai.py  # Hyperscale — 150 agents, 793 KB payloads
```

The headline result: under 50 concurrent agents, single-writer SQLite meets the
50 ms SLA on **~0.4%** of writes (disk lock contention), while swapping to Redis
takes SLA compliance to **100%** with a **p99 of ~1 ms** — no core code changes.
See [`benchmarks/README.md`](benchmarks/README.md) for the full analysis.

The overhead budget is enforced *in code*: a checkpoint write that would exceed it
is dropped and logged as *missed* rather than ever blocking your agent thread.

## How it works

```
ExecutionNode ──► CheckpointStore (Tier 2: durable, append-only)
      ▲                  ▲
      │                  │
 Adapters          CheckpointEngine ──► HotCache (Tier 1: LRU + TTL)
 (LangGraph/             │
  CrewAI/           RecoveryEngine ──► RecoveryPlan (replay vs. skip)
  OpenAI)           ReplaySession  ──► FULL / FROM_NODE / MOCK_TOOLS / COUNTERFACTUAL
```

Every execution is a DAG of `ExecutionNode` records. The log is never mutated,
only appended to — so any point in time can be reconstructed deterministically.
`TOOL` nodes default to **non-idempotent**, which is how recovery knows never to
re-run side effects. See [docs/concepts.md](docs/concepts.md) for the full model.

### The Dual-Tier Cache

Recovery reads have to be instant, but memory can't grow without bound. So the
engine keeps checkpoints in two tiers:

- **Tier 1 — Hot cache (RAM).** An in-process LRU cache with per-entry TTL holds
  the most recent checkpoints. Recovery reads hit it in **microseconds** and it
  is updated *synchronously* on every write, so the latest state is always
  available even if the durable write is dropped.
- **Tier 2 — Cold store (durable).** Any `CheckpointStore` (SQLite by default,
  Redis or PostgreSQL optional) holds the full append-only history. Reads that
  miss the hot cache fall back here and re-populate Tier 1.

A read checks Tier 1 first (a hit is ~4 µs), then Tier 2 (~190 µs) — a **~40×**
speedup for the common case of recovering a run that just crashed.

### The 50 ms SLA Budget

Checkpointing sits on the hot path of every agent step, so it must never stall the
agent. The engine enforces a hard **overhead budget** (default 50 ms) *in code*:

```python
try:
    await asyncio.wait_for(self.store.write(node), timeout=self._budget_seconds)
except asyncio.TimeoutError:
    self.metrics.increment("checkpoint.timeout")   # logged as "missed"
    return False                                   # agent continues, never blocked
```

If a durable write would exceed the budget, it is **dropped and recorded as
missed** rather than blocking execution. Because Tier 1 already holds the state,
recovery is unaffected. This is why swapping a lock-contended SQLite store for
Redis takes SLA compliance from ~0.4% to 100% under load — the budget protects the
agent, and a faster backend simply lets more writes land durably.

## Showcase: drop-in LangGraph Redis saver

LangGraph's built-in `SqliteSaver` is **blocking and single-writer**. Replace it
with a **non-blocking, horizontally-scalable** saver in one line:

```diff
- from langgraph.checkpoint.sqlite import SqliteSaver
- saver = SqliteSaver.from_conn_string("agent.db")
+ from showcase.langgraph_redis_saver import LivingAIRedisSaver
+ saver = LivingAIRedisSaver.from_url("redis://localhost:6379")

  graph = builder.compile(checkpointer=saver)
```

A **runnable** demo (no LangGraph or Redis server needed) shows crash recovery
skipping a non-idempotent tool call:

```bash
pip install "livingai[redis]" fakeredis
python showcase/demo_langgraph_redis.py
```

See [`showcase/README.md`](showcase/README.md) for the full comparison table.

## Living AI Studio (visual dashboard)

A CLI tells you *what* happened; **Studio shows you** — and lets you rewind. It
renders every execution as an interactive graph (successes green, failures red,
side-effect nodes as boxes) with a **"Replay from this node"** button.

```bash
pip install "livingai[studio]"
python studio/seed_demo.py
streamlit run studio/app.py -- --db studio_demo.db
```

See [`studio/README.md`](studio/README.md).

## CLI

```bash
livingai list   --db agent.db          # execution ids
livingai show   run-1 --db agent.db    # the node graph
livingai replay run-1 --db agent.db --mode MOCK_TOOLS
```

## Documentation

[Quickstart](docs/quickstart.md) ·
[Concepts](docs/concepts.md) ·
[Checkpointing](docs/checkpointing.md) ·
[Recovery](docs/recovery.md) ·
[Replay](docs/replay.md) ·
[CLI](docs/cli.md) ·
[Adapters](docs/adapters.md) ·
[Migrating from other checkpointers](docs/migration.md) ·
[API Reference](docs/api-reference.md)

## Quality

- **128 tests, 100% line coverage** — including crash-simulation and stress tests
  (10k-node graphs, concurrent writers, write contention).
- **`mypy --strict` clean** across all source files; ships `py.typed`.
- **CI matrix** on Python 3.9–3.12 with a 100%-coverage gate.

```bash
pip install -e ".[dev]"
python -m pytest -q                    # run the suite
mypy --strict livingai                 # type check
python benchmarks/benchmark.py         # reproduce the numbers above
```

## Design principles

| Principle | How |
| --- | --- |
| Zero-dependency core | Standard library only (`sqlite3`, `asyncio`, `zlib`, `dataclasses`, `uuid`). |
| Append-only log | Every write inserts a new row; nothing is mutated or deleted. |
| Framework-agnostic | No framework imports in the core; framework data lives in `metadata`. |
| Async-first I/O | Storage is `async`; sync SQLite runs off the event loop. |
| Bounded overhead | Cold writes run under `asyncio.wait_for`; overruns are dropped, never blocking the agent. |

## Roadmap

Shipped: core data model, checkpoint engine, recovery engine, replay engine, CLI,
LangGraph / CrewAI / OpenAI adapters, three-tier benchmark suite, docs, Redis
store, PostgreSQL store, drop-in LangGraph Redis saver, and the visual Studio
dashboard.

**Optional backends** — swap the default SQLite store for Redis or PostgreSQL
with a single import (no core changes required):

```bash
pip install "livingai[redis]"     # hot Redis store
pip install "livingai[postgres]"  # PostgreSQL cold store
```

```python
from livingai.stores.redis import RedisStore
from livingai.stores.postgres import PostgresStore

# Redis
engine = CheckpointEngine(RedisStore(url="redis://localhost:6379"))

# PostgreSQL
store = PostgresStore(dsn="postgresql://user:pass@localhost/livingai")
await store.initialize()          # creates tables once
engine = CheckpointEngine(store)
```

A **Docker Compose** dev stack (Postgres + Redis) ships with the repo:

```bash
docker compose up -d    # starts postgres:5432 + redis:6379
```

Next: FastAPI cloud backend (5 endpoints), cloud client (`CloudSync`), hosted Studio.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, running tests, code
style, and how to add a new framework adapter or storage backend.

## License

Apache-2.0 — see [LICENSE](LICENSE).
