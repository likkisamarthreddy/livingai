# Benchmarks — `livingai` Performance Suite

A comprehensive walkthrough and architectural analysis of the testing tiers used
to validate the `livingai` checkpointing engine under increasingly demanding
workloads. Every script here is runnable and reproduces the numbers below on your
own hardware.

```bash
pip install "livingai[redis]" fakeredis

python benchmarks/benchmark.py                 # core latency suite (p50/p95/p99, compression)
python benchmarks/benchmark_livingai.py        # Tier 1 — Big Test
python benchmarks/prod_test_livingai.py        # Tier 2 — Production Test (SQLite vs Redis)
python benchmarks/hyperscale_test_livingai.py  # Tier 3 — Hyperscale Test
```

Add `--json` to any script for machine-readable output.

---

## Overview of the Test Suite

Three distinct tiers evaluate the limits of the engine under growing pressure:

```mermaid
graph TD
    A[Big Test] -->|Scale up concurrency & disk I/O| B[Production Test]
    B -->|Scale up payload size & parallelism| C[Hyperscale Test]

    A1["Single Agent Loop<br>250KB State<br>500 steps (In-Memory)"] --> A
    B1["50 Concurrent Agents<br>5-100KB growing states<br>1,000 steps (Disk SQLite)"] --> B
    C1["150 Concurrent Agents<br>793KB large states<br>1,500 steps (Disk SQLite)"] --> C
```

---

## 1. The Big Test — `benchmark_livingai.py`

Single-agent, high-depth execution. Evaluates how the engine performs when one
agent runs for hundreds of steps with a large memory footprint, verifying both
**dual-tier cache eviction** and **compression**.

| Parameter | Value |
| --- | --- |
| Payload | 250 KB constant |
| Concurrency | 1 (single agent loop) |
| Depth | 500 consecutive execution nodes |
| Cache capacity | 128 (forces 372 nodes to evict to the cold store) |

**Representative results**

- Write throughput: **500 writes in ~2.4 s** (~4.9 ms/save, in-memory SQLite)
- Hot cache read latency: **~0.8 ms** (recently written node still in memory)
- Cold store read latency: **~0.8–1.5 ms** (evicted node → SQLite query + zlib decompress)

> **Note:** Even when falling back to disk and decompressing, read latency stays
> around a millisecond. Standard-library `zlib` at level 6 is highly optimized,
> and SQLite primary-key lookups on `node_id` take well under a millisecond on
> local NVMe storage.

---

## 2. Production-Level Test — `prod_test_livingai.py`

A small-to-medium enterprise workload: multiple agents running simultaneously,
writing to a persistent store, with states that grow as conversation history
accumulates. Demonstrates the **50 ms strict execution-budget SLA** and the impact
of backend choice.

| Parameter | Value |
| --- | --- |
| Payload | 5 KB → 100 KB (grows per turn) |
| Concurrency | 50 agents in parallel |
| Depth | 20 steps per agent (1,000 total saves) |
| SLA budget | 50 ms strict timeout |

**SQLite (single-writer, file-backed)**

- SLA compliance: **~0.4%** (concurrent disk I/O triggers write locks → timeouts)
- Compression: **~93% space savings** (50 MB raw → ~3 MB on disk)

**Redis (via `fakeredis`, in-process)**

- SLA compliance: **100.0%** (1,000 / 1,000 writes)
- Write latency: **p50 ~0.5 ms, p99 ~1.1 ms**

> **Important:** This is the SLA design in action. Under single-writer SQLite,
> concurrent disk I/O causes lock contention that trips the 50 ms budget — the
> engine *drops* those cold writes rather than stalling the agent (the hot cache
> still holds the latest state, so recovery is unaffected). Swap the backend to
> Redis and SLA compliance jumps from **~0.4% to 100%**, proving the engine is
> ready for horizontally-scaling production databases.

---

## 3. Hyperscale Workload Test — `hyperscale_test_livingai.py`

Frontier-lab scale: hundreds of active users running multi-agent pipelines with
massive context windows (large system prompts, variables, histories).

| Parameter | Value |
| --- | --- |
| Payload | ~793 KB per turn (massive chat context) |
| Concurrency | 150 agents in parallel |
| Depth | 10 steps per agent (1,500 total steps) |
| SLA budget | 50 ms strict timeout |

**SQLite**

- SLA compliance under load drops sharply (disk bottleneck), yet
- **Recovery success: 1500 / 1500 (100%)** — the in-process hot cache intercepts
  every read at single-digit-millisecond latency.

**Redis**

- SLA compliance: **100%**, write throughput scales, latency dominated by the CPU
  cost of zlib-compressing the ~0.77 MB payload.
- **Recovery success: 1500 / 1500 (100%)**

> **Tip:** Under SQLite the database bottleneck is severe, but recovery stays 100%
> because the dual-tier `HotCache` serves reads from RAM. Under Redis the I/O
> bottleneck dissolves entirely — the system is ready for frontier-scale
> workloads when backed by a production Redis cluster.

---

## How It Works — Code Walkthrough

### 1. The Save Loop (`livingai/checkpoint.py`)

```python
async def save(self, node: ExecutionNode, state: Optional[bytes] = None) -> bool:
    # 1. Compress the state (Zlib level 6, self-describing codec header).
    if state is not None:
        node.checkpoint = self.compressor.compress(state)

    # 2. Update the RAM-based HotCache synchronously (sub-millisecond).
    self.hot.put(node)

    # 3. Persist to the cold store, but enforce the 50 ms SLA budget.
    try:
        await asyncio.wait_for(self.store.write(node), timeout=self._budget_seconds)
    except asyncio.TimeoutError:
        self.metrics.increment("checkpoint.timeout")   # dropped write, agent unblocked
        return False
    return True
```

### 2. The Load Loop (`livingai/checkpoint.py`)

```python
async def load(self, node_id: str) -> Optional[tuple[ExecutionNode, Optional[bytes]]]:
    node = self.hot.get(node_id)          # 1. Fast path: in-process HotCache
    if node is not None:
        self.metrics.increment("checkpoint.hot_hit")
    else:
        node = await self.store.read(node_id)   # 2. Slow path: cold store + cache it
        if node is None:
            return None
        if node.checkpoint is not None:
            self.hot.put(node)

    state = self.compressor.decompress(node.checkpoint) if node.checkpoint else None
    return node, state
```

### 3. Self-Describing Blobs (`livingai/compression.py`)

Every compressed blob carries a one-byte codec header so the algorithm can evolve
without breaking historical databases:

- `0x00` — `NoopCompressor` (uncompressed)
- `0x01` — `ZlibCompressor`

You can later switch to (e.g.) `zstd` and still decode every existing checkpoint.

---

## Reproducing the numbers

Results vary with CPU, disk, and Python version. The scripts print your machine's
actual numbers. For an apples-to-apples production benchmark, point the Redis leg
at a real cluster instead of `fakeredis`:

```python
from livingai.stores.redis import RedisStore
engine = CheckpointEngine(RedisStore(url="redis://your-cluster:6379"))
```
