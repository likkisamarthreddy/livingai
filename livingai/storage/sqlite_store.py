"""Zero-config SQLite checkpoint store.

This is the default store: ``SQLiteStore()`` works immediately with no
configuration, persisting to an on-disk (or in-memory) SQLite database. It is
the reference implementation that every other backend is tested against.

**Append-only semantics.** Each :meth:`write` inserts a *new* row keyed by an
auto-incrementing ``seq``. Prior rows are never updated or deleted, so the full
history of every node transition is preserved and crash-safe. Reads return the
projection with the highest ``seq`` for a given node id.

All public methods are async. Synchronous SQLite calls are dispatched to a
thread so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from typing import Optional

from ..graph import ExecutionNode


__all__ = ["SQLiteStore"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    data         TEXT NOT NULL,
    checkpoint   BLOB
);
CREATE INDEX IF NOT EXISTS idx_nodes_id ON nodes (id);
CREATE INDEX IF NOT EXISTS idx_nodes_execution ON nodes (execution_id);
"""


class SQLiteStore:
    """SQLite-backed :class:`~livingai.storage.CheckpointStore` implementation.

    Args:
        path: Database file path. Defaults to an in-memory database, which is
            ideal for tests and ephemeral runs. Pass a file path for durable
            local persistence (the zero-config production-local default).
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        # check_same_thread=False because asyncio.to_thread may run calls on
        # different worker threads; a process-wide lock serializes access.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- sync helpers (run inside a worker thread) -------------------------

    def _write_sync(self, node: ExecutionNode) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO nodes (id, execution_id, data, checkpoint) "
                "VALUES (?, ?, ?, ?)",
                (node.id, node.execution_id, node.to_json(), node.checkpoint),
            )
            self._conn.commit()

    def _read_sync(self, node_id: str) -> Optional[ExecutionNode]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data, checkpoint FROM nodes WHERE id = ? "
                "ORDER BY seq DESC LIMIT 1",
                (node_id,),
            ).fetchone()
        return self._row_to_node(row)

    def _list_sync(self, execution_id: str) -> list[ExecutionNode]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT data, checkpoint FROM nodes n
                WHERE execution_id = ?
                  AND seq = (
                      SELECT MAX(seq) FROM nodes m WHERE m.id = n.id
                  )
                ORDER BY (SELECT MIN(seq) FROM nodes o WHERE o.id = n.id)
                """,
                (execution_id,),
            ).fetchall()
        nodes = [self._row_to_node(r) for r in rows]
        return [n for n in nodes if n is not None]

    def _latest_checkpoint_sync(
        self, execution_id: str
    ) -> Optional[ExecutionNode]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data, checkpoint FROM nodes "
                "WHERE execution_id = ? AND checkpoint IS NOT NULL "
                "ORDER BY seq DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
        return self._row_to_node(row)

    @staticmethod
    def _row_to_node(row: Optional[sqlite3.Row]) -> Optional[ExecutionNode]:
        if row is None:
            return None
        checkpoint = row["checkpoint"]
        if checkpoint is not None:
            checkpoint = bytes(checkpoint)
        return ExecutionNode.from_json(row["data"], checkpoint=checkpoint)

    # -- async protocol methods -------------------------------------------

    async def write(self, node: ExecutionNode) -> None:
        await asyncio.to_thread(self._write_sync, node)

    async def read(self, node_id: str) -> Optional[ExecutionNode]:
        return await asyncio.to_thread(self._read_sync, node_id)

    async def list_by_execution(self, execution_id: str) -> list[ExecutionNode]:
        return await asyncio.to_thread(self._list_sync, execution_id)

    async def get_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[ExecutionNode]:
        return await asyncio.to_thread(self._latest_checkpoint_sync, execution_id)

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()
