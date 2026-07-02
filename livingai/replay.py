"""Replay engine — Phase 3.

Replay is distinct from recovery. *Recovery* is automatic and happens on crash.
*Replay* is manual and happens for debugging, testing, and optimization. It
re-runs a previously recorded execution from the durable log.

Four modes (per the Technical Execution Plan):

* ``FULL`` — re-execute every node from scratch.
* ``FROM_NODE`` — re-execute from a specific ``node_id`` onward.
* ``MOCK_TOOLS`` — re-execute, but tool calls return their **recorded** output
  instead of hitting real APIs. This is the most valuable debugging mode: it
  lets you re-run LLM reasoning against the exact same tool responses without
  triggering real side effects.
* ``COUNTERFACTUAL`` — re-execute with a modified input at a specific node to ask
  "what would have happened if…".

The engine never mutates the stored log; it reads recorded nodes and drives a
caller-supplied async ``handler`` that performs the actual (re-)execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from .graph import ExecutionNode, NodeType
from .metrics import Metrics
from .storage import CheckpointStore


__all__ = ["ReplayMode", "ReplayResult", "ReplaySession"]

logger = logging.getLogger("livingai.replay")

ReplayHandler = Callable[[ExecutionNode], Awaitable[Any]]


class ReplayMode(str, Enum):
    FULL = "FULL"
    FROM_NODE = "FROM_NODE"
    MOCK_TOOLS = "MOCK_TOOLS"
    COUNTERFACTUAL = "COUNTERFACTUAL"


@dataclass
class ReplayResult:
    """Outcome of replaying a single node."""

    node: ExecutionNode
    output: Any
    mocked: bool = False


class ReplaySession:
    """Replays a recorded execution in one of the :class:`ReplayMode` modes."""

    def __init__(
        self,
        store: CheckpointStore,
        execution_id: str,
        *,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.store = store
        self.execution_id = execution_id
        self.metrics = metrics or Metrics()

    async def run(
        self,
        handler: ReplayHandler,
        *,
        mode: ReplayMode = ReplayMode.FULL,
        from_node_id: Optional[str] = None,
        counterfactual: Optional[tuple[str, Any]] = None,
    ) -> list[ReplayResult]:
        """Replay the execution and return per-node results.

        Args:
            handler: Async callable invoked to (re-)execute a node. Receives the
                recorded :class:`ExecutionNode` (with a possibly overridden
                ``input`` under COUNTERFACTUAL) and returns its output.
            mode: One of the :class:`ReplayMode` values.
            from_node_id: Required for ``FROM_NODE``; the node to start from.
            counterfactual: Required for ``COUNTERFACTUAL``; a
                ``(node_id, new_input)`` tuple.
        """
        if mode is ReplayMode.FROM_NODE and not from_node_id:
            raise ValueError("FROM_NODE mode requires from_node_id")
        if mode is ReplayMode.COUNTERFACTUAL and counterfactual is None:
            raise ValueError("COUNTERFACTUAL mode requires counterfactual=(node_id, input)")

        nodes = await self.store.list_by_execution(self.execution_id)
        nodes = self._select(nodes, mode, from_node_id)

        cf_node_id, cf_input = counterfactual if counterfactual else (None, None)

        results: list[ReplayResult] = []
        for node in nodes:
            if mode is ReplayMode.MOCK_TOOLS and node.type is NodeType.TOOL:
                # Return the recorded output — no real tool call.
                results.append(ReplayResult(node=node, output=node.output, mocked=True))
                self.metrics.increment("replay.mocked")
                continue

            if mode is ReplayMode.COUNTERFACTUAL and node.id == cf_node_id:
                # Replay this node with a modified input.
                probe = ExecutionNode.from_dict(node.to_dict())
                probe.input = cf_input
                output = await handler(probe)
                results.append(ReplayResult(node=probe, output=output))
            else:
                output = await handler(node)
                results.append(ReplayResult(node=node, output=output))

            self.metrics.increment("replay.executed")

        self.metrics.increment("replay.sessions")
        logger.info(
            "Replayed %s in %s mode: %d nodes", self.execution_id, mode.value, len(results)
        )
        return results

    @staticmethod
    def _select(
        nodes: list[ExecutionNode],
        mode: ReplayMode,
        from_node_id: Optional[str],
    ) -> list[ExecutionNode]:
        if mode is ReplayMode.FROM_NODE:
            index = next((i for i, n in enumerate(nodes) if n.id == from_node_id), None)
            if index is None:
                raise KeyError(f"from_node_id not found in execution: {from_node_id}")
            return nodes[index:]
        return nodes
