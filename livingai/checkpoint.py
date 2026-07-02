"""Checkpoint engine — Phase 1.

Responsibilities:

* **Write** a checkpoint: compress the state blob and persist it to the durable
  (cold) store, while enforcing the 50 ms overhead budget.
* **Read** a checkpoint back, transparently decompressing it.
* **Evict** stale checkpoints from the hot tier (LRU + TTL), keeping recovery
  reads fast without unbounded memory growth.

Overhead budget is enforced *in code*, not merely measured: the store write runs
under :func:`asyncio.wait_for`. If it exceeds the budget the checkpoint is
recorded as *missed* and execution continues unblocked — a checkpoint is never
allowed to stall the agent thread.

The engine owns two tiers, as described in the plan:

* **Tier 1 — hot cache**: an in-process LRU+TTL cache of the most recent
  checkpoints, sized for sub-millisecond recovery reads.
* **Tier 2 — cold store**: any :class:`~livingai.storage.CheckpointStore`
  (SQLite by default), holding the durable append-only history.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Optional

from .compression import Compressor, ZlibCompressor
from .graph import ExecutionNode
from .metrics import Metrics
from .storage import CheckpointStore


__all__ = ["CheckpointEngine", "HotCache"]

logger = logging.getLogger("livingai.checkpoint")


class HotCache:
    """Bounded LRU cache with per-entry TTL for the latest checkpoints.

    Keys are ``node_id``; values are :class:`ExecutionNode` instances carrying a
    checkpoint blob.
    """

    def __init__(self, capacity: int = 128, ttl_seconds: float = 3600.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, tuple[float, ExecutionNode]]" = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def put(self, node: ExecutionNode) -> None:
        now = self._now()
        if node.id in self._entries:
            self._entries.move_to_end(node.id)
        self._entries[node.id] = (now, node)
        self._entries.move_to_end(node.id)
        self._evict_expired(now)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)  # evict least-recently-used

    def get(self, node_id: str) -> Optional[ExecutionNode]:
        entry = self._entries.get(node_id)
        if entry is None:
            return None
        ts, node = entry
        if self._now() - ts > self.ttl_seconds:
            del self._entries[node_id]
            return None
        self._entries.move_to_end(node_id)  # mark as recently used
        return node

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, (ts, _) in self._entries.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self._entries[k]

    def __len__(self) -> int:
        self._evict_expired(self._now())
        return len(self._entries)


class CheckpointEngine:
    """Compresses, budgets, caches, and persists execution checkpoints.

    Args:
        store: Durable (cold) checkpoint store.
        overhead_budget_ms: Hard p99 write budget. Writes exceeding it are
            dropped and logged as missed (default 50 ms).
        compressor: Codec for checkpoint blobs (default zlib).
        hot_capacity: Max entries retained in the hot tier.
        hot_ttl_seconds: TTL for hot-tier entries.
        metrics: Optional shared metrics sink.
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        overhead_budget_ms: float = 50.0,
        compressor: Optional[Compressor] = None,
        hot_capacity: int = 128,
        hot_ttl_seconds: float = 3600.0,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.store = store
        self.overhead_budget_ms = overhead_budget_ms
        self.compressor = compressor or ZlibCompressor()
        self.hot = HotCache(capacity=hot_capacity, ttl_seconds=hot_ttl_seconds)
        self.metrics = metrics or Metrics()

    @property
    def _budget_seconds(self) -> float:
        return self.overhead_budget_ms / 1000.0

    async def save(self, node: ExecutionNode, state: Optional[bytes] = None) -> bool:
        """Persist a checkpoint, enforcing the overhead budget.

        If ``state`` is provided it is compressed into ``node.checkpoint``;
        otherwise any pre-set ``node.checkpoint`` bytes are used as-is (they are
        assumed already encoded by this engine's compressor).

        Returns ``True`` if the durable write completed within budget, ``False``
        if it was dropped as a missed checkpoint. Never raises on timeout.
        """
        if state is not None:
            node.checkpoint = self.compressor.compress(state)

        # Hot tier is updated synchronously and cheaply — it is the recovery
        # fast-path and must reflect the latest state even if the cold write is
        # dropped.
        self.hot.put(node)

        start = time.monotonic()
        try:
            await asyncio.wait_for(self.store.write(node), timeout=self._budget_seconds)
        except asyncio.TimeoutError:
            self.metrics.increment("checkpoint.timeout")
            logger.warning("Checkpoint missed (budget exceeded): %s", node.id)
            return False
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            self.metrics.observe("checkpoint.write_ms", elapsed_ms)

        self.metrics.increment("checkpoint.write")
        return True

    async def load(self, node_id: str) -> Optional[tuple[ExecutionNode, Optional[bytes]]]:
        """Return ``(node, decompressed_state)`` for ``node_id``.

        Checks the hot tier first, then falls back to the durable store. Returns
        ``None`` if the node is unknown.
        """
        node = self.hot.get(node_id)
        if node is not None:
            self.metrics.increment("checkpoint.hot_hit")
        else:
            node = await self.store.read(node_id)
            if node is None:
                return None
            self.metrics.increment("checkpoint.cold_hit")
            if node.checkpoint is not None:
                self.hot.put(node)

        state = (
            self.compressor.decompress(node.checkpoint)
            if node.checkpoint is not None
            else None
        )
        return node, state

    async def latest(
        self, execution_id: str
    ) -> Optional[tuple[ExecutionNode, Optional[bytes]]]:
        """Return the most recent checkpoint for an execution, decompressed."""
        node = await self.store.get_latest_checkpoint(execution_id)
        if node is None:
            return None
        if node.checkpoint is not None:
            self.hot.put(node)
        state = (
            self.compressor.decompress(node.checkpoint)
            if node.checkpoint is not None
            else None
        )
        return node, state
