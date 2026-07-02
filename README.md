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
2000 writes — the same configuration you get out of the box. Reproduce with
`python benchmarks/benchmark.py`.

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

- **108 tests, 100% line coverage** — including crash-simulation and stress tests
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
LangGraph / CrewAI / OpenAI adapters, benchmarks, docs, Redis store, PostgreSQL store.

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

Next: FastAPI cloud backend (5 endpoints), cloud client (`CloudSync`), web replay dashboard.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, running tests, code
style, and how to add a new framework adapter or storage backend.

## License

Apache-2.0 — see [LICENSE](LICENSE).
