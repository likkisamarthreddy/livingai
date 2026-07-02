"""Unit tests for the checkpoint engine, compression, hot cache, and metrics."""

import asyncio
import time

import pytest

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    HotCache,
    Metrics,
    NoopCompressor,
    NodeType,
    SQLiteStore,
    ZlibCompressor,
)
from livingai.compression import _CODEC_ZLIB


def run(coro):
    return asyncio.run(coro)


def make_node(execution_id="run-1", **kwargs):
    return ExecutionNode(execution_id=execution_id, type=NodeType.PROMPT, **kwargs)


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------

def test_zlib_round_trip():
    c = ZlibCompressor()
    data = b"hello world " * 100
    assert c.decompress(c.compress(data)) == data


def test_zlib_reduces_size_on_repetitive_data():
    c = ZlibCompressor()
    data = b"A" * 10_000
    compressed = c.compress(data)
    assert len(compressed) < len(data) * 0.2  # >80% reduction


def test_zlib_has_codec_header():
    c = ZlibCompressor()
    assert c.compress(b"x")[0] == _CODEC_ZLIB


def test_noop_round_trip():
    c = NoopCompressor()
    assert c.decompress(c.compress(b"abc")) == b"abc"


def test_cross_codec_decompress():
    # A blob written by zlib can be decoded by the noop codec and vice versa,
    # because the codec byte is self-describing.
    z = ZlibCompressor()
    n = NoopCompressor()
    blob = z.compress(b"payload data " * 20)
    assert n.decompress(blob) == b"payload data " * 20


def test_empty_blob_decompresses_to_empty():
    assert ZlibCompressor().decompress(b"") == b""


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        ZlibCompressor(level=99)


def test_unknown_codec_raises():
    with pytest.raises(ValueError):
        ZlibCompressor().decompress(b"\xff garbage")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_increment():
    m = Metrics()
    m.increment("a")
    m.increment("a", 2)
    assert m.counter("a") == 3


def test_metrics_percentile():
    m = Metrics()
    for v in range(1, 101):
        m.observe("lat", v)
    assert m.percentile("lat", 50) == pytest.approx(50.5)
    assert m.percentile("lat", 99) == pytest.approx(99.01, abs=0.5)


def test_metrics_percentile_empty():
    assert Metrics().percentile("none", 99) is None


def test_metrics_single_sample():
    m = Metrics()
    m.observe("x", 7.0)
    assert m.percentile("x", 99) == 7.0


def test_metrics_samples_and_snapshot():
    m = Metrics()
    m.increment("c")
    m.observe("s", 1.0)
    m.observe("s", 2.0)
    assert m.samples("s") == [1.0, 2.0]
    snap = m.snapshot()
    assert snap["counters"]["c"] == 1
    assert snap["samples"]["s"] == [1.0, 2.0]


# --------------------------------------------------------------------------
# HotCache
# --------------------------------------------------------------------------

def test_hot_cache_put_get():
    cache = HotCache(capacity=4)
    node = make_node()
    cache.put(node)
    assert cache.get(node.id) is node


def test_hot_cache_lru_eviction():
    cache = HotCache(capacity=2)
    a, b, c = make_node(), make_node(), make_node()
    cache.put(a)
    cache.put(b)
    cache.get(a.id)       # a is now most-recently-used
    cache.put(c)          # evicts b (LRU)
    assert cache.get(a.id) is a
    assert cache.get(b.id) is None
    assert cache.get(c.id) is c


def test_hot_cache_ttl_expiry():
    cache = HotCache(capacity=8, ttl_seconds=0.05)
    node = make_node()
    cache.put(node)
    assert cache.get(node.id) is node
    time.sleep(0.06)
    assert cache.get(node.id) is None


def test_hot_cache_capacity_validation():
    with pytest.raises(ValueError):
        HotCache(capacity=0)


def test_hot_cache_len_excludes_expired():
    cache = HotCache(capacity=8, ttl_seconds=0.05)
    cache.put(make_node())
    time.sleep(0.06)
    assert len(cache) == 0


# --------------------------------------------------------------------------
# CheckpointEngine
# --------------------------------------------------------------------------

@pytest.fixture
def engine():
    store = SQLiteStore()
    yield CheckpointEngine(store)
    store.close()


def test_save_compresses_and_persists(engine):
    node = make_node()
    state = b"conversation history " * 100
    ok = run(engine.save(node, state=state))
    assert ok is True
    # Stored blob is compressed, not the raw state.
    assert node.checkpoint is not None and node.checkpoint != state
    assert engine.metrics.counter("checkpoint.write") == 1


def test_load_from_hot_tier(engine):
    node = make_node()
    state = b"state-payload"
    run(engine.save(node, state=state))
    result = run(engine.load(node.id))
    assert result is not None
    loaded_node, loaded_state = result
    assert loaded_state == state
    assert engine.metrics.counter("checkpoint.hot_hit") == 1


def test_load_from_cold_store():
    store = SQLiteStore()
    try:
        # Fresh engine with tiny hot cache to force a cold read on a second engine.
        writer = CheckpointEngine(store)
        node = make_node()
        run(writer.save(node, state=b"durable-state"))

        reader = CheckpointEngine(store)  # empty hot cache
        result = run(reader.load(node.id))
        assert result is not None
        _, state = result
        assert state == b"durable-state"
        assert reader.metrics.counter("checkpoint.cold_hit") == 1
    finally:
        store.close()


def test_load_missing_returns_none(engine):
    assert run(engine.load("does-not-exist")) is None


def test_latest_checkpoint(engine):
    n1 = make_node(execution_id="run-X")
    run(engine.save(n1, state=b"first"))
    n2 = make_node(execution_id="run-X")
    run(engine.save(n2, state=b"second"))
    result = run(engine.latest("run-X"))
    assert result is not None
    _, state = result
    assert state == b"second"


def test_latest_none_when_no_checkpoint(engine):
    run(engine.save(make_node(execution_id="run-Y")))  # no state
    assert run(engine.latest("run-Y")) is None


def test_save_without_state_uses_existing_blob(engine):
    # Pre-encoded blob passed directly on the node.
    blob = ZlibCompressor().compress(b"already-encoded")
    node = make_node(checkpoint=blob)
    ok = run(engine.save(node))
    assert ok is True
    _, state = run(engine.load(node.id))
    assert state == b"already-encoded"


# --- Overhead budget enforcement -----------------------------------------

class SlowStore:
    """A store whose write always exceeds any reasonable budget."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.writes = 0

    async def write(self, node):
        self.writes += 1
        await asyncio.sleep(self.delay)

    async def read(self, node_id):
        return None

    async def list_by_execution(self, execution_id):
        return []

    async def get_latest_checkpoint(self, execution_id):
        return None


def test_budget_exceeded_returns_false_and_counts_miss():
    slow = SlowStore(delay=0.2)
    engine = CheckpointEngine(slow, overhead_budget_ms=10)
    node = make_node()
    ok = run(engine.save(node, state=b"x"))
    assert ok is False
    assert engine.metrics.counter("checkpoint.timeout") == 1
    assert engine.metrics.counter("checkpoint.write") == 0


def test_budget_miss_still_updates_hot_tier():
    slow = SlowStore(delay=0.2)
    engine = CheckpointEngine(slow, overhead_budget_ms=10)
    node = make_node()
    run(engine.save(node, state=b"recover-me"))
    # Even though the cold write was dropped, recovery can read from hot tier.
    result = run(engine.load(node.id))
    assert result is not None
    _, state = result
    assert state == b"recover-me"


def test_budget_records_write_latency_sample(engine):
    run(engine.save(make_node(), state=b"y"))
    assert engine.metrics.percentile("checkpoint.write_ms", 99) is not None
