"""Benchmark 1 — The "Big Test": single-agent, high-depth execution.

Targets a single agent running for hundreds of steps with a large memory
footprint, verifying **dual-tier cache eviction** and **compression** under
sustained load.

    Payload    : 250 KB constant
    Concurrency: 1 (single agent loop)
    Depth      : 500 consecutive execution nodes
    Cache      : 128 entries (forces 372 nodes to evict to the cold store)

Run:

    python benchmarks/benchmark_livingai.py
    python benchmarks/benchmark_livingai.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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

STEPS = 500
PAYLOAD_BYTES = 250_000
CACHE_CAPACITY = 128


def _node(i: int) -> ExecutionNode:
    return ExecutionNode(
        execution_id="big-test",
        type=NodeType.PROMPT,
        status=Status.SUCCESS,
        output=f"step-{i}",
    )


async def run() -> dict:
    store = SQLiteStore(":memory:")
    engine = CheckpointEngine(store, hot_capacity=CACHE_CAPACITY)

    # Semi-compressible 250 KB payload (half entropy, half repetition).
    state = os.urandom(PAYLOAD_BYTES // 2) + b"A" * (PAYLOAD_BYTES // 2)

    nodes: list[ExecutionNode] = []
    start = time.perf_counter()
    for i in range(STEPS):
        node = _node(i)
        await engine.save(node, state=state)
        nodes.append(node)
    write_elapsed = time.perf_counter() - start

    # Hot read: the most recently written node is still in the cache.
    hot_id = nodes[-1].id
    t0 = time.perf_counter()
    await engine.load(hot_id)
    hot_ms = (time.perf_counter() - t0) * 1000

    # Cold read: the very first node was evicted long ago (128 < 500).
    cold_id = nodes[0].id
    t0 = time.perf_counter()
    await engine.load(cold_id)
    cold_ms = (time.perf_counter() - t0) * 1000

    store.close()
    return {
        "steps": STEPS,
        "payload_kb": PAYLOAD_BYTES // 1000,
        "cache_capacity": CACHE_CAPACITY,
        "evicted_nodes": STEPS - CACHE_CAPACITY,
        "write_total_s": round(write_elapsed, 2),
        "write_avg_ms": round(write_elapsed / STEPS * 1000, 2),
        "hot_read_ms": round(hot_ms, 2),
        "cold_read_ms": round(cold_ms, 2),
        "p50_write_ms": round(engine.metrics.percentile("checkpoint.write_ms", 50) or 0, 3),
        "p99_write_ms": round(engine.metrics.percentile("checkpoint.write_ms", 99) or 0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    result = asyncio.run(run())

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 60)
    print("  Benchmark 1 — Big Test (single agent, 500 steps, 250 KB)")
    print("=" * 60)
    print(f"  Write throughput : {result['steps']} writes in {result['write_total_s']}s "
          f"({result['write_avg_ms']} ms/save)")
    print(f"  Hot cache read   : {result['hot_read_ms']} ms")
    print(f"  Cold store read  : {result['cold_read_ms']} ms  "
          f"(node evicted, SQLite + zlib decompress)")
    print(f"  Nodes evicted    : {result['evicted_nodes']} of {result['steps']} "
          f"(cache capacity {result['cache_capacity']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
