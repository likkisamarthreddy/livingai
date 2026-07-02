"""Recovery engine — Phase 2.

When a process crashes and restarts, the recovery engine reconstructs execution
state from the last durable checkpoint and determines how to resume. This is the
core value proposition of the runtime.

Recovery flow (per the Technical Execution Plan):

1. Runtime asks: does ``execution_id`` have a checkpoint?
2. If yes:
   a. Load the latest checkpoint (hot tier, then cold store).
   b. Deserialize the execution state.
   c. Replay **idempotent** nodes recorded after the checkpoint.
   d. Resume execution from the crash point.
3. If no: start a fresh execution.

**Critical constraint.** Tool calls with external side effects (API writes,
emails, payments) are *non-idempotent* and must never be re-executed during
recovery. Idempotency is decided by :meth:`ExecutionNode.is_idempotent`; a
framework adapter is responsible for annotating tool calls correctly.

The engine is deliberately split into two concerns:

* :meth:`RecoveryEngine.plan` — pure analysis. Reads the durable log and returns
  an immutable :class:`RecoveryPlan`. No side effects, easy to test.
* :meth:`RecoveryEngine.replay` — drives a caller-supplied async handler over
  the plan's replayable nodes, skipping the non-idempotent ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .checkpoint import CheckpointEngine
from .graph import ExecutionNode, Status
from .metrics import Metrics


__all__ = ["RecoveryPlan", "RecoveryEngine"]

logger = logging.getLogger("livingai.recovery")

NodeHandler = Callable[[ExecutionNode], Awaitable[object]]


@dataclass(frozen=True)
class RecoveryPlan:
    """The result of analysing an execution for recovery.

    Attributes:
        execution_id: The execution analysed.
        found: Whether a durable checkpoint was found. If ``False`` the caller
            should start a fresh execution and every other field is empty.
        checkpoint_node: The node carrying the checkpoint to resume from.
        state: The decompressed checkpoint state bytes (may be ``None``).
        replay_nodes: Idempotent nodes recorded after the checkpoint, in
            execution order — safe to re-run.
        skipped_nodes: Non-idempotent nodes recorded after the checkpoint — must
            NOT be re-run (their side effects already happened).
    """

    execution_id: str
    found: bool
    checkpoint_node: Optional[ExecutionNode] = None
    state: Optional[bytes] = None
    replay_nodes: list[ExecutionNode] = field(default_factory=list)
    skipped_nodes: list[ExecutionNode] = field(default_factory=list)

    @property
    def resume_node_id(self) -> Optional[str]:
        return self.checkpoint_node.id if self.checkpoint_node else None


class RecoveryEngine:
    """Reconstructs and resumes crashed executions from durable checkpoints."""

    def __init__(
        self,
        checkpoint_engine: CheckpointEngine,
        *,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.checkpoints = checkpoint_engine
        self.store = checkpoint_engine.store
        self.metrics = metrics or checkpoint_engine.metrics

    async def plan(self, execution_id: str) -> RecoveryPlan:
        """Analyse an execution and produce a :class:`RecoveryPlan`.

        Never mutates state. Nodes recorded *after* the checkpoint node (by
        execution/creation order) are partitioned into replayable (idempotent)
        and skipped (non-idempotent) buckets.
        """
        latest = await self.checkpoints.latest(execution_id)
        if latest is None:
            self.metrics.increment("recovery.fresh_start")
            logger.info("No checkpoint for %s — fresh start", execution_id)
            return RecoveryPlan(execution_id=execution_id, found=False)

        checkpoint_node, state = latest
        nodes = await self.store.list_by_execution(execution_id)

        # Everything strictly after the checkpoint node in execution order is a
        # candidate for replay. list_by_execution is ordered by first appearance.
        after = _nodes_after(nodes, checkpoint_node.id)

        replay_nodes: list[ExecutionNode] = []
        skipped_nodes: list[ExecutionNode] = []
        for node in after:
            if node.is_idempotent():
                replay_nodes.append(node)
            else:
                skipped_nodes.append(node)

        self.metrics.increment("recovery.planned")
        logger.info(
            "Recovery plan for %s: resume=%s replay=%d skipped=%d",
            execution_id,
            checkpoint_node.id,
            len(replay_nodes),
            len(skipped_nodes),
        )
        return RecoveryPlan(
            execution_id=execution_id,
            found=True,
            checkpoint_node=checkpoint_node,
            state=state,
            replay_nodes=replay_nodes,
            skipped_nodes=skipped_nodes,
        )

    async def replay(self, plan: RecoveryPlan, handler: NodeHandler) -> list[Any]:
        """Re-run the plan's replayable nodes via an async ``handler``.

        The handler is invoked once per idempotent node, in order. Non-idempotent
        nodes are skipped (counted, never invoked). Returns the list of handler
        results for the replayed nodes.
        """
        if not plan.found:
            return []

        results = []
        for node in plan.replay_nodes:
            result = await handler(node)
            results.append(result)
            self.metrics.increment("recovery.replayed")
        for _ in plan.skipped_nodes:
            self.metrics.increment("recovery.skipped")
        return results

    async def recover(self, execution_id: str, handler: NodeHandler) -> RecoveryPlan:
        """Convenience: :meth:`plan` then :meth:`replay` in one call.

        Returns the plan that was executed (its ``found`` flag tells the caller
        whether recovery happened or a fresh start is required).
        """
        plan = await self.plan(execution_id)
        await self.replay(plan, handler)
        return plan


def _nodes_after(nodes: list[ExecutionNode], node_id: str) -> list[ExecutionNode]:
    """Return the nodes that appear after ``node_id`` in the given ordered list.

    If ``node_id`` is not present (e.g. the checkpoint node was never listed),
    no nodes are considered "after" it and an empty list is returned.
    """
    index = next((i for i, n in enumerate(nodes) if n.id == node_id), None)
    if index is None:
        return []
    return nodes[index + 1 :]
