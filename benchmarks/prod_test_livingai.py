"""Benchmark 2 — Production-Level Test: concurrent agents, SLA budget, SQLite vs Redis.

Simulates a small-to-medium enterprise workload: many agents running in parallel,
writing to a persistent store, with states that grow as conversation history
accumulates. Demonstrates the **50 ms strict execution-budget SLA** and how the
backend choice (single-writer SQLite vs horizontally-scaled Redis) governs
compliance.

    Payload    : 5 KB -> 100 KB (grows per turn)
    Concurrency: 50 agents in parallel
    Depth      : 20 steps per agent (1,000 total saves)
    SLA budget : 50 ms strict timeout

The Redis leg uses ``fakeredis`` (an in-process Redis) so it runs with no server.
Swap it for a real ``redis://`` URL to benchmark a production cluster.

Run:

    pip install livingai[redis] fakeredis
    python benchmarks/prod_test_livingai.py
    python benchmarks/prod_test_livingai.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time

# The SLA budget intentionally drops slow cold-store writes; silence the
# per-miss warnings so the benchmark summary stays readable.
logging.getLogger("livingai.checkpoint").setLevel(logging.ERROR)

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    SQLiteStore,
    Status,
)

AGENTS = 50
STEPS_PER_AGENT = 20
MIN_PAYLOAD = 5_000
MAX_PAYLOAD = 100_000


def _payload(step: int) -> bytes:
    size = MIN_PAYLOAD + int((MAX_PAYLOAD - MIN_PAYLOAD) * (step / STEPS_PER_AGENT))
    return os.urandom(size // 2) + b"A" * (size // 2)


async def _agent(engine: CheckpointEngine, agent_id: int) -> tuple[int, int, int]:
    ok = missed = raw_bytes = 0
    for step in range(STEPS_PER_AGENT):
        state = _payload(step)
        raw_bytes += len(state)
        node = ExecutionNode(
            execution_id=f"agent-{agent_id}",
            type=NodeType.PROMPT,
            status=Status.SUCCESS,
            output=f"turn-{step}",
        )
        if await engine.save(node, state=state):
            ok += 1
        else:
            missed += 1
    return ok, missed, raw_bytes


async def _run_backend(engine: CheckpointEngine) -> dict:
    start = time.perf_counter()
    results = await asyncio.gather(
        *(_agent(engine, a) for a in range(AGENTS))
    )
    elapsed = time.perf_counter() - start

    ok = sum(r[0] for r in results)
    missed = sum(r[1] for r in results)
    raw = sum(r[2] for r in results)
    total = ok + missed
    return {
        "successful_writes": ok,
        "missed_writes": missed,
        "sla_compliance_pct": round(ok / total * 100, 2),
        "elapsed_s": round(elapsed, 2),
        "raw_mb": round(raw / 1_000_000, 2),
        "p50_ms": round(engine.metrics.percentile("checkpoint.write_ms", 50) or 0, 2),
        "p95_ms": round(engine.metrics.percentile("checkpoint.write_ms", 95) or 0, 2),
        "p99_ms": round(engine.metrics.percentile("checkpoint.write_ms", 99) or 0, 2),
    }


async def run() -> dict:
    # --- SQLite (file-backed, single-writer) ---
    db_path = tempfile.mktemp(suffix=".db")
    sqlite_store = SQLiteStore(db_path)
    sqlite_engine = CheckpointEngine(sqlite_store)
    sqlite = await _run_backend(sqlite_engine)
    disk_bytes = os.path.getsize(db_path)
    sqlite["disk_mb"] = round(disk_bytes / 1_000_000, 2)
    sqlite["compression_pct"] = round((1 - disk_bytes / (sqlite["raw_mb"] * 1_000_000)) * 100, 2)
    sqlite_store.close()
    os.unlink(db_path)

    # --- Redis (via fakeredis, in-process) ---
    redis = {"skipped": "install fakeredis + livingai[redis] to run the Redis leg"}
    try:
        import fakeredis.aioredis
        from livingai.stores.redis import RedisStore

        client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_engine = CheckpointEngine(RedisStore(client=client))
        redis = await _run_backend(redis_engine)
    except ImportError:
        pass

    return {"sqlite": sqlite, "redis": redis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run())

    if args.json:
        print(json.dumps(result, indent=2))
        return

    s = result["sqlite"]
    print("=" * 62)
    print("  Benchmark 2 — Production Test (50 agents, 1,000 saves, 50ms SLA)")
    print("=" * 62)
    print("  [SQLite — single-writer, file-backed]")
    print(f"    SLA compliance : {s['sla_compliance_pct']}%  "
          f"({s['successful_writes']} ok / {s['missed_writes']} missed)")
    print(f"    Elapsed        : {s['elapsed_s']}s")
    print(f"    Raw data       : {s['raw_mb']} MB -> disk {s['disk_mb']} MB "
          f"({s['compression_pct']}% saved)")
    r = result["redis"]
    if "skipped" in r:
        print("  [Redis] skipped:", r["skipped"])
    else:
        print("  [Redis — fakeredis, in-process]")
        print(f"    SLA compliance : {r['sla_compliance_pct']}%  "
              f"({r['successful_writes']} ok / {r['missed_writes']} missed)")
        print(f"    Elapsed        : {r['elapsed_s']}s")
        print(f"    Write latency  : p50 {r['p50_ms']}ms  p95 {r['p95_ms']}ms  p99 {r['p99_ms']}ms")
    print("=" * 62)


if __name__ == "__main__":
    main()
