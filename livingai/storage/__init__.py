"""Storage layer for the Living AI runtime.

The storage layer is defined by a single :class:`CheckpointStore` protocol so any
backend (SQLite, Redis, Postgres, S3, ...) is interchangeable and passes the same
test suite. The core runtime depends only on this protocol, never on a concrete
backend.

All I/O is async-first, per the SDK design contract.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..graph import ExecutionNode

from .sqlite_store import SQLiteStore


__all__ = ["CheckpointStore", "SQLiteStore"]


@runtime_checkable
class CheckpointStore(Protocol):
    """Interface implementable for any storage backend.

    Implementations must be **append-only**: a ``write`` never mutates or deletes
    prior records. Reads return the latest projection of a node.
    """

    async def write(self, node: ExecutionNode) -> None:
        """Append a node (or a new state of an existing node) to the log."""
        ...

    async def read(self, node_id: str) -> Optional[ExecutionNode]:
        """Return the latest state of ``node_id``, or ``None`` if unknown."""
        ...

    async def list_by_execution(self, execution_id: str) -> list[ExecutionNode]:
        """Return the latest state of every node in an execution, oldest first."""
        ...

    async def get_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[ExecutionNode]:
        """Return the most recent node carrying a checkpoint blob, if any."""
        ...
