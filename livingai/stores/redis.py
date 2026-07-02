"""Redis-backed checkpoint store.

Install the optional dependency::

    pip install livingai[redis]

Usage::

    from livingai.stores.redis import RedisStore
    from livingai import CheckpointEngine

    store = RedisStore(url="redis://localhost:6379")
    engine = CheckpointEngine(store)

**Design.**
Each ``ExecutionNode`` is stored as a JSON string in a Redis Hash keyed by
``node_id``. The checkpoint blob (bytes) is stored in a parallel Hash keyed by
``{node_id}:ckpt``. Append-only semantics are implemented with a sorted set per
``execution_id`` that keeps members ``node_id`` scored by their first-seen
timestamp, guaranteeing stable insertion-order enumeration.

Three Redis keys per execution:

* ``lai:n:{node_id}``           — latest node JSON (string)
* ``lai:c:{node_id}``           — latest checkpoint blob (bytes, may be missing)
* ``lai:e:{execution_id}``      — sorted set: node_id → first-seen epoch

This intentionally keeps the key layout flat (no nested hashes) so TTL policies
and inspection work on predictable key patterns.

All I/O is async; it runs through the ``redis.asyncio`` client so it never blocks
the event loop.
"""

from __future__ import annotations

import time
from typing import Optional

from ..graph import ExecutionNode

try:
    import redis.asyncio as aioredis  # type: ignore[import-not-found,import-untyped]
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Redis support requires 'redis>=4.2'. Install with: pip install livingai[redis]"
    ) from e


__all__ = ["RedisStore"]

_N = "lai:n:"   # node JSON prefix
_C = "lai:c:"   # checkpoint blob prefix
_E = "lai:e:"   # execution sorted-set prefix


class RedisStore:
    """Redis-backed :class:`~livingai.storage.CheckpointStore`.

    Args:
        url:    Redis URL, e.g. ``redis://localhost:6379``.
        ttl:    Seconds before node keys expire. ``None`` means never (default).
        client: Inject a pre-built ``redis.asyncio.Redis`` client (useful for
                tests with ``fakeredis``).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        ttl: Optional[int] = None,
        client: Optional[object] = None,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._client: aioredis.Redis = (  # type: ignore[type-arg]
            client  # type: ignore[assignment]
            if client is not None
            else aioredis.from_url(url, decode_responses=False)  # pragma: no cover
        )

    async def close(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    # -- CheckpointStore protocol ------------------------------------------

    async def write(self, node: ExecutionNode) -> None:
        pipe = self._client.pipeline(transaction=False)

        # Latest node state
        pipe.set(_N + node.id, node.to_json().encode())
        if self._ttl:
            pipe.expire(_N + node.id, self._ttl)

        # Checkpoint blob (absent if none)
        if node.checkpoint is not None:
            pipe.set(_C + node.id, node.checkpoint)
            if self._ttl:
                pipe.expire(_C + node.id, self._ttl)

        # Execution index: NX keeps the *first* score (insertion order)
        score = time.time()
        pipe.zadd(_E + node.execution_id, {node.id: score}, nx=True)

        await pipe.execute()

    async def read(self, node_id: str) -> Optional[ExecutionNode]:
        raw, blob = await self._client.mget(_N + node_id, _C + node_id)
        if raw is None:
            return None
        return ExecutionNode.from_json(raw.decode(), checkpoint=blob or None)

    async def list_by_execution(
        self, execution_id: str
    ) -> list[ExecutionNode]:
        node_ids: list[bytes] = await self._client.zrange(
            _E + execution_id, 0, -1
        )
        if not node_ids:
            return []
        keys = [_N + nid.decode() for nid in node_ids]
        ckpt_keys = [_C + nid.decode() for nid in node_ids]
        raws = await self._client.mget(*keys)
        blobs = await self._client.mget(*ckpt_keys)
        nodes = []
        for raw, blob in zip(raws, blobs):
            if raw is None:  # pragma: no cover
                continue
            nodes.append(
                ExecutionNode.from_json(raw.decode(), checkpoint=blob or None)
            )
        return nodes

    async def get_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[ExecutionNode]:
        """Return the most recent node carrying a checkpoint blob."""
        node_ids: list[bytes] = await self._client.zrevrange(
            _E + execution_id, 0, -1
        )
        for nid in node_ids:
            blob: Optional[bytes] = await self._client.get(_C + nid.decode())
            if blob is not None:
                raw = await self._client.get(_N + nid.decode())
                if raw is None:  # pragma: no cover
                    continue
                return ExecutionNode.from_json(raw.decode(), checkpoint=blob)
        return None
