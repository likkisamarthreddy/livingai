"""Comprehensive benchmarks for the Living AI runtime.

Publishes the numbers the Technical Execution Plan calls for, beyond just
checkpoint write latency:

* checkpoint write latency (p50/p95/p99)
* recovery time vs log size
* hot vs cold read performance
* compression ratio across payload profiles
* storage overhead per 1,000-node execution

Run with:

    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    RecoveryEngine,
    SQLiteStore,
    Status,
    ZlibCompressor,
)


def _node(execution_id="bench", ntype=NodeType.PROMPT, **kw):
    return ExecutionNode(execution_id=execution_id, type=ntype, status=Status.SUCCESS, **kw)


async def bench_write_latency(n=2000, blob_size=50_000):
    store = SQLiteStore()
    engine = CheckpointEngine(store)
    state = os.urandom(blob_size // 2) + b"A" * (blob_size // 2)  # semi-compressible
    for _ in range(n):
        await engine.save(_node(), state=state)
    m = engine.metrics
    store.close()
    return {
        "samples": n,
        "blob_bytes": blob_size,
        "p50_ms": round(m.percentile("checkpoint.write_ms", 50), 3),
        "p95_ms": round(m.percentile("checkpoint.write_ms", 95), 3),
        "p99_ms": round(m.percentile("checkpoint.write_ms", 99), 3),
        "missed": m.counter("checkpoint.timeout"),
    }


async def bench_recovery_vs_log_size(sizes=(100, 1000, 5000)):
    results = []
    for size in sizes:
        store = SQLiteStore()
        engine = CheckpointEngine(store)
        recovery = RecoveryEngine(engine)
        await engine.save(_node(), state=b"snapshot")
        for _ in range(size):
            await engine.save(_node(ntype=NodeType.PROMPT))
        start = time.perf_counter()
        plan = await recovery.plan("bench")
        elapsed_ms = (time.perf_counter() - start) * 1000
        store.close()
        results.append({"log_size": size, "plan_ms": round(elapsed_ms, 3),
                        "replay_nodes": len(plan.replay_nodes)})
    return results


async def bench_hot_vs_cold(n=1000):
    store = SQLiteStore()
    engine = CheckpointEngine(store, hot_capacity=n)
    nodes = [_node() for _ in range(n)]
    for node in nodes:
        await engine.save(node, state=b"x" * 1000)

    # Hot reads (all in cache).
    start = time.perf_counter()
    for node in nodes:
        await engine.load(node.id)
    hot_ms = (time.perf_counter() - start) * 1000 / n

    # Cold reads: fresh engine, empty hot cache.
    cold_engine = CheckpointEngine(store, hot_capacity=1)
    start = time.perf_counter()
    for node in nodes:
        await cold_engine.load(node.id)
    cold_ms = (time.perf_counter() - start) * 1000 / n
    store.close()
    return {"reads": n, "hot_us": round(hot_ms * 1000, 2), "cold_us": round(cold_ms * 1000, 2)}


def bench_compression():
    c = ZlibCompressor()
    profiles = {
        "repetitive": b"A" * 100_000,
        "json_like": (b'{"role":"user","content":"hello world"}' * 2000),
        "random": os.urandom(100_000),
    }
    out = {}
    for name, data in profiles.items():
        comp = c.compress(data)
        out[name] = {
            "raw_bytes": len(data),
            "compressed_bytes": len(comp),
            "ratio": round(1 - len(comp) / len(data), 3),
        }
    return out


async def bench_storage_overhead(n=1000):
    path = os.path.join(tempfile.gettempdir(), "livingai_bench.db")
    if os.path.exists(path):
        os.remove(path)
    store = SQLiteStore(path)
    engine = CheckpointEngine(store)
    for i in range(n):
        await engine.save(_node(input={"i": i}, output={"r": i}), state=b"s" * 500)
    store.close()
    size = os.path.getsize(path)
    os.remove(path)
    return {"nodes": n, "db_bytes": size, "bytes_per_node": round(size / n, 1)}


async def main_async():
    return {
        "write_latency": await bench_write_latency(),
        "recovery_vs_log_size": await bench_recovery_vs_log_size(),
        "hot_vs_cold_read": await bench_hot_vs_cold(),
        "compression": bench_compression(),
        "storage_overhead": await bench_storage_overhead(),
    }


def _print_human(r):
    w = r["write_latency"]
    print("Checkpoint write latency (50KB blob):")
    print(f"  p50 {w['p50_ms']}ms  p95 {w['p95_ms']}ms  p99 {w['p99_ms']}ms  missed={w['missed']}")
    print("\nRecovery planning time vs log size:")
    for row in r["recovery_vs_log_size"]:
        print(f"  {row['log_size']:>5} nodes -> {row['plan_ms']}ms ({row['replay_nodes']} replayable)")
    hc = r["hot_vs_cold_read"]
    print(f"\nRead latency: hot {hc['hot_us']}us vs cold {hc['cold_us']}us per read")
    print("\nCompression ratio by payload:")
    for name, c in r["compression"].items():
        print(f"  {name:<11} {c['raw_bytes']:>7}B -> {c['compressed_bytes']:>7}B ({c['ratio']:.0%} saved)")
    so = r["storage_overhead"]
    print(f"\nStorage: {so['bytes_per_node']}B per node ({so['db_bytes']} bytes / {so['nodes']} nodes)")


def main():
    parser = argparse.ArgumentParser(description="Living AI runtime benchmarks")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    results = asyncio.run(main_async())
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
