# Living AI Runtime Documentation

Checkpoint, recovery, and replay infrastructure for AI agents — a zero-dependency
core that makes agent executions crash-safe, resumable, and debuggable.

## Contents

- [Quickstart](quickstart.md) — install and run your first checkpointed execution.
- [Concepts](concepts.md) — the execution graph, append-only log, and storage tiers.
- [Checkpointing](checkpointing.md) — compression, the hot tier, and the 50 ms budget.
- [Recovery](recovery.md) — crash recovery and idempotent replay.
- [Replay](replay.md) — the four replay modes (incl. `MOCK_TOOLS`).
- [CLI](cli.md) — the `livingai` command-line tool.
- [Adapters](adapters.md) — integrating frameworks (LangGraph, CrewAI, OpenAI Agents).
- [Migration & Integration](migration.md) — adopting the runtime alongside your stack.
- [API Reference](api-reference.md) — the public surface.

## Design principles

| Principle | What it means |
| --- | --- |
| Zero-dependency core | The runtime uses only the Python standard library. |
| Append-only log | Executions are never mutated, only appended to — crash-safe by construction. |
| Framework-agnostic | The core knows nothing about LangGraph/CrewAI; adapters translate. |
| Async-first I/O | All storage is `async`; blocking work runs off the event loop. |
| 50 ms overhead budget | Checkpoint writes that exceed budget are dropped, never blocking the agent. |

## Project status

Phases 0–4 are implemented: data model, checkpoint engine, recovery engine,
replay engine, CLI, and open-source packaging. See the
[repository README](../README.md) for the phase-by-phase status and benchmarks.
