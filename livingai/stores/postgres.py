"""PostgreSQL-backed checkpoint store.

Install the optional dependency::

    pip install livingai[postgres]

Usage::

    from livingai.stores.postgres import PostgresStore
    from livingai import CheckpointEngine

    store = PostgresStore(dsn="postgresql://user:pass@localhost/db")
    await store.initialize()          # creates tables if they don't exist
    engine = CheckpointEngine(store)

**Design.**
An append-only ``nodes`` table where each row is an immutable *version* of a
node. Reads use a window function to project the latest version per node id.
The schema mirrors the SQLite store exactly so the two backends can be swapped
transparently.

Schema::

    CREATE TABLE IF NOT EXISTS lai_nodes (
        seq          BIGSERIAL PRIMARY KEY,
        id           TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        data         TEXT NOT NULL,
        checkpoint   BYTEA
    );

Indices on ``id`` and ``execution_id`` match the SQLite store.

All I/O is async, using the ``asyncpg`` driver.
"""

from __future__ import annotations

from typing import Optional

from ..graph import ExecutionNode

try:
    import asyncpg  # type: ignore[import-not-found,import-untyped]
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PostgreSQL support requires 'asyncpg'. Install with: pip install livingai[postgres]"
    ) from e


__all__ = ["PostgresStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lai_nodes (
    seq          BIGSERIAL PRIMARY KEY,
    id           TEXT        NOT NULL,
    execution_id TEXT        NOT NULL,
    data         TEXT        NOT NULL,
    checkpoint   BYTEA
);
CREATE INDEX IF NOT EXISTS lai_idx_node_id  ON lai_nodes (id);
CREATE INDEX IF NOT EXISTS lai_idx_exec_id  ON lai_nodes (execution_id);
"""

# Latest-version projection: for each unique node_id keep the row with the
# highest seq, ordered by the *first* seq that node_id appeared.
_LIST_SQL = """
SELECT DISTINCT ON (id)
       data, checkpoint
FROM   lai_nodes
WHERE  execution_id = $1
ORDER  BY id, seq DESC
"""

_LIST_ORDER_SQL = """
SELECT data, checkpoint FROM (
    SELECT DISTINCT ON (id)
           data, checkpoint,
           MIN(seq) OVER (PARTITION BY id) AS first_seq
    FROM   lai_nodes
    WHERE  execution_id = $1
    ORDER  BY id, seq DESC
) sub
ORDER BY first_seq
"""

_LATEST_CKPT_SQL = """
SELECT DISTINCT ON (id)
       data, checkpoint
FROM   lai_nodes
WHERE  execution_id = $1
  AND  checkpoint IS NOT NULL
ORDER  BY id, seq DESC
"""


class PostgresStore:
    """asyncpg-backed :class:`~livingai.storage.CheckpointStore`.

    Args:
        dsn:  PostgreSQL DSN, e.g.
              ``postgresql://user:pass@localhost/livingai``.
        pool: Inject a pre-built ``asyncpg.Pool`` (useful for tests).
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/livingai",
        *,
        pool: Optional[object] = None,
    ) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = pool  # type: ignore[type-arg]

    async def initialize(self) -> None:  # pragma: no cover
        """Create tables and indices if they don't exist. Call once on startup."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()

    # -- CheckpointStore protocol ------------------------------------------

    async def write(self, node: ExecutionNode) -> None:
        assert self._pool is not None, "Call initialize() before write()"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO lai_nodes (id, execution_id, data, checkpoint) "
                "VALUES ($1, $2, $3, $4)",
                node.id,
                node.execution_id,
                node.to_json(),
                node.checkpoint,
            )

    async def read(self, node_id: str) -> Optional[ExecutionNode]:
        assert self._pool is not None, "Call initialize() before read()"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data, checkpoint FROM lai_nodes "
                "WHERE id = $1 ORDER BY seq DESC LIMIT 1",
                node_id,
            )
        return _row_to_node(row)

    async def list_by_execution(
        self, execution_id: str
    ) -> list[ExecutionNode]:
        assert self._pool is not None, "Call initialize() before list_by_execution()"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_LIST_ORDER_SQL, execution_id)
        nodes = [_row_to_node(r) for r in rows]
        return [n for n in nodes if n is not None]

    async def get_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[ExecutionNode]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_LATEST_CKPT_SQL, execution_id)
        # rows are latest-per-node; return the one with the highest global seq
        # by fetching seq alongside data.
        if not rows:  # pragma: no cover
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT DISTINCT ON (id) data, checkpoint "
                "FROM lai_nodes "
                "WHERE execution_id = $1 AND checkpoint IS NOT NULL "
                "ORDER BY id, seq DESC, seq DESC "
                "LIMIT 1",
                execution_id,
            )
        return _row_to_node(row)


def _row_to_node(row: Optional[object]) -> Optional[ExecutionNode]:  # pragma: no cover
    if row is None:
        return None
    return ExecutionNode.from_json(
        row["data"],  # type: ignore[index]
        checkpoint=row["checkpoint"],  # type: ignore[index]
    )
