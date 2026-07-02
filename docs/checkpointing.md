# Checkpointing

The `CheckpointEngine` compresses agent state, writes it through a hot tier to a
durable store, and enforces a hard overhead budget.

```python
from livingai import CheckpointEngine, SQLiteStore

engine = CheckpointEngine(
    SQLiteStore("agent.db"),
    overhead_budget_ms=50,     # hard p99 write budget
    hot_capacity=128,          # Tier-1 LRU size
    hot_ttl_seconds=3600,      # Tier-1 entry TTL
)
```

## Saving

```python
ok = await engine.save(node, state=b"...serialized agent state...")
```

- If `state` is provided it is compressed into `node.checkpoint`.
- The hot tier is updated synchronously (the recovery fast-path).
- The durable write runs under the overhead budget. Returns `True` on success,
  `False` if the write was dropped as *missed*.

## The 50 ms overhead budget

The budget is enforced **in code**, not merely measured:

```python
try:
    await asyncio.wait_for(store.write(node), timeout=0.050)
except asyncio.TimeoutError:
    metrics.increment("checkpoint.timeout")   # logged as missed
    return False                              # execution continues unblocked
```

A slow storage backend can never stall the agent thread. Missed checkpoints are
counted (`checkpoint.timeout`) and the hot tier still holds the latest state, so
recovery remains possible.

## Compression

Checkpoints are compressed before storage. The default `ZlibCompressor` uses the
standard library, keeping the core dependency-free. Every blob carries a one-byte
codec header, so payloads are self-describing and future codecs (e.g. zstd) can
coexist with old data.

```python
from livingai import ZlibCompressor, NoopCompressor

engine = CheckpointEngine(store, compressor=ZlibCompressor(level=9))
```

Typical reduction on repetitive agent state (histories, retrieved documents) is
60–80%+.

## Loading

```python
result = await engine.load(node_id)   # (node, state) or None
node, state = result

latest = await engine.latest("run-1") # most recent checkpoint for an execution
```

`load` checks the hot tier first (`checkpoint.hot_hit`) then the cold store
(`checkpoint.cold_hit`).

## Metrics

The engine records counters and latency samples on a `Metrics` object:

| Metric | Meaning |
| --- | --- |
| `checkpoint.write` | Successful durable writes |
| `checkpoint.timeout` | Missed checkpoints (budget exceeded) |
| `checkpoint.write_ms` | Per-write latency samples (percentiles) |
| `checkpoint.hot_hit` / `checkpoint.cold_hit` | Load tier hits |

```python
p99 = engine.metrics.percentile("checkpoint.write_ms", 99)
```
