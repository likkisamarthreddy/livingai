"""Shared base for framework adapters.

All adapters translate a framework's node lifecycle (start / end / error) into
runtime :class:`ExecutionNode` records persisted through a
:class:`CheckpointEngine`. The only per-framework differences are:

* ``framework`` — the tag stored in ``metadata["framework"]``.
* ``node_key`` — the metadata key holding the framework's node name.
* ``tool_hints`` — substrings that classify a node name as a side-effecting
  ``TOOL`` (and therefore non-idempotent) node.

Concrete adapters subclass :class:`BaseAdapter` and set those three attributes.
"""

from __future__ import annotations

from typing import Any, Optional

from ..checkpoint import CheckpointEngine
from ..graph import ErrorInfo, ExecutionNode, NodeType, Status, new_id, utcnow


__all__ = ["BaseAdapter", "DEFAULT_TOOL_HINTS", "classify_name"]


DEFAULT_TOOL_HINTS: tuple[str, ...] = (
    "tool",
    "action",
    "call",
    "api",
    "http",
    "search",
    "fetch",
    "write",
)


def classify_name(
    name: str,
    tool_hints: tuple[str, ...],
    *,
    node_type: Optional[NodeType] = None,
) -> NodeType:
    """Map a framework node name to a :class:`NodeType`.

    An explicit ``node_type`` always wins; otherwise the name is matched against
    ``tool_hints``, defaulting to ``PROMPT``.
    """
    if node_type is not None:
        return node_type
    lowered = name.lower()
    if any(hint in lowered for hint in tool_hints):
        return NodeType.TOOL
    return NodeType.PROMPT


class BaseAdapter:
    """Common node-lifecycle recording shared by all framework adapters."""

    framework: str = "base"
    node_key: str = "node"
    tool_hints: tuple[str, ...] = DEFAULT_TOOL_HINTS

    def __init__(
        self,
        engine: CheckpointEngine,
        execution_id: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.execution_id = execution_id or new_id()

    def classify(self, name: str, *, node_type: Optional[NodeType] = None) -> NodeType:
        return classify_name(name, self.tool_hints, node_type=node_type)

    async def on_node_start(
        self,
        name: str,
        *,
        input: Any = None,
        node_type: Optional[NodeType] = None,
        parent_id: Optional[str] = None,
        idempotent: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionNode:
        """Create and persist a RUNNING node for a starting framework step."""
        ntype = self.classify(name, node_type=node_type)
        meta: dict[str, Any] = {"framework": self.framework, self.node_key: name}
        if metadata:
            meta.update(metadata)
        if idempotent is not None:
            meta["idempotent"] = idempotent

        node = ExecutionNode(
            execution_id=self.execution_id,
            type=ntype,
            parent_id=parent_id,
            status=Status.RUNNING,
            input=input,
            metadata=meta,
        )
        await self.engine.save(node)
        return node

    async def on_node_end(
        self,
        node: ExecutionNode,
        *,
        output: Any = None,
        state: Optional[bytes] = None,
        cost_tokens: Optional[int] = None,
    ) -> bool:
        """Mark a node SUCCESS and optionally attach a checkpoint blob."""
        node.status = Status.SUCCESS
        node.output = output
        node.completed_at = utcnow()
        node.cost_tokens = cost_tokens
        self._set_latency(node)
        return await self.engine.save(node, state=state)

    async def on_node_error(
        self,
        node: ExecutionNode,
        error: BaseException,
    ) -> bool:
        """Mark a node FAILED with structured error info."""
        node.status = Status.FAILED
        node.completed_at = utcnow()
        node.error = ErrorInfo(type=type(error).__name__, message=str(error))
        self._set_latency(node)
        return await self.engine.save(node)

    @staticmethod
    def _set_latency(node: ExecutionNode) -> None:
        if node.completed_at and node.created_at:
            delta = node.completed_at - node.created_at
            node.latency_ms = int(delta.total_seconds() * 1000)
