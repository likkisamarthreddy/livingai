"""Execution graph data model for the Living AI runtime.

This module defines the foundational data structure that everything else in the
runtime is built on: the :class:`ExecutionNode`. Every agent execution produces
a directed acyclic graph (DAG) of these nodes.

Design principles (see the Technical Execution Plan):

* **Append-only** — nodes describe events in an execution log. State is modelled
  as a sequence of node writes, never in-place mutation.
* **Framework agnostic** — this module has zero dependencies on any agent
  framework. Framework-specific data lives in the free-form ``metadata`` field.
* **Schema-stable** — ``input`` / ``output`` / ``metadata`` are JSON blobs, not
  typed columns, so the schema stays stable across framework and product
  versions. Schema evolution is an application-layer concern.

The module depends only on the Python standard library.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


__all__ = [
    "NodeType",
    "Status",
    "ErrorInfo",
    "ExecutionNode",
    "new_id",
    "utcnow",
]


def new_id() -> str:
    """Return a fresh globally-unique identifier as a string."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class NodeType(str, Enum):
    """The kind of work a node represents."""

    PROMPT = "PROMPT"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    BRANCH = "BRANCH"


class Status(str, Enum):
    """Lifecycle status of a node."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class ErrorInfo:
    """Structured error attached to a failed node."""

    type: str
    message: str
    traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["ErrorInfo"]:
        if data is None:
            return None
        return cls(
            type=data["type"],
            message=data["message"],
            traceback=data.get("traceback"),
        )


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


@dataclass
class ExecutionNode:
    """A single node in an execution graph.

    ``input`` / ``output`` / ``metadata`` are stored as plain JSON-serializable
    values. ``checkpoint`` is an opaque (optionally compressed) byte blob holding
    serialized agent state used by the recovery engine.
    """

    execution_id: str
    type: NodeType
    id: str = field(default_factory=new_id)
    parent_id: Optional[str] = None
    status: Status = Status.PENDING
    created_at: datetime = field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
    input: Any = None
    output: Any = None
    error: Optional[ErrorInfo] = None
    cost_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[bytes] = None

    def __post_init__(self) -> None:
        # Accept raw strings for enum fields for ergonomic construction and
        # round-tripping from storage.
        if not isinstance(self.type, NodeType):
            self.type = NodeType(self.type)
        if not isinstance(self.status, Status):
            self.status = Status(self.status)

    # -- semantics ---------------------------------------------------------

    def is_idempotent(self) -> bool:
        """Whether re-executing this node during recovery is safe.

        Precedence:

        1. An explicit ``metadata["idempotent"]`` boolean always wins — this is
           how a framework adapter annotates a specific tool call.
        2. Otherwise ``TOOL`` nodes default to **non-idempotent** (they may have
           external side effects: API writes, emails, payments), while all other
           node types default to idempotent (safe to replay).
        """
        flag = self.metadata.get("idempotent")
        if isinstance(flag, bool):
            return flag
        return self.type is not NodeType.TOOL


    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict.

        The ``checkpoint`` blob is intentionally excluded because it is binary.
        Storage backends persist it separately. Use :meth:`to_row` for a
        representation that carries the checkpoint bytes.
        """
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "execution_id": self.execution_id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": _to_iso(self.created_at),
            "completed_at": _to_iso(self.completed_at),
            "input": self.input,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "cost_tokens": self.cost_tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], checkpoint: Optional[bytes] = None
    ) -> "ExecutionNode":
        return cls(
            execution_id=data["execution_id"],
            type=NodeType(data["type"]),
            id=data["id"],
            parent_id=data.get("parent_id"),
            status=Status(data["status"]),
            created_at=_from_iso(data.get("created_at")) or utcnow(),
            completed_at=_from_iso(data.get("completed_at")),
            input=data.get("input"),
            output=data.get("output"),
            error=ErrorInfo.from_dict(data.get("error")),
            cost_tokens=data.get("cost_tokens"),
            latency_ms=data.get("latency_ms"),
            metadata=data.get("metadata") or {},
            checkpoint=checkpoint,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string (excludes the binary checkpoint blob)."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, checkpoint: Optional[bytes] = None) -> "ExecutionNode":
        return cls.from_dict(json.loads(raw), checkpoint=checkpoint)
