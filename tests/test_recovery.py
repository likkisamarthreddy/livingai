"""Recovery engine tests, including crash simulation."""

import asyncio

import pytest

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    RecoveryEngine,
    SQLiteStore,
    Status,
)
from livingai.recovery import _nodes_after


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def rig():
    store = SQLiteStore()
    engine = CheckpointEngine(store)
    recovery = RecoveryEngine(engine)
    yield engine, recovery
    store.close()


def node(execution_id, ntype=NodeType.PROMPT, **kw):
    return ExecutionNode(execution_id=execution_id, type=ntype, **kw)


# --- idempotency semantics ------------------------------------------------

def test_prompt_is_idempotent():
    assert node("e").is_idempotent() is True


def test_tool_default_non_idempotent():
    assert node("e", NodeType.TOOL).is_idempotent() is False


def test_metadata_override_wins():
    assert node("e", NodeType.TOOL, metadata={"idempotent": True}).is_idempotent() is True
    assert node("e", NodeType.PROMPT, metadata={"idempotent": False}).is_idempotent() is False


# --- planning -------------------------------------------------------------

def test_nodes_after_missing_id_returns_empty():
    nodes = [node("e"), node("e")]
    assert _nodes_after(nodes, "not-present") == []


def test_plan_fresh_start_when_no_checkpoint(rig):
    engine, recovery = rig
    run(engine.save(node("run-1")))  # node without checkpoint
    plan = run(recovery.plan("run-1"))
    assert plan.found is False
    assert plan.checkpoint_node is None
    assert plan.replay_nodes == []
    assert engine.metrics.counter("recovery.fresh_start") == 1


def test_plan_resumes_from_latest_checkpoint(rig):
    engine, recovery = rig
    a = node("run-1")
    run(engine.save(a, state=b"snapshot-A"))
    plan = run(recovery.plan("run-1"))
    assert plan.found is True
    assert plan.resume_node_id == a.id
    assert plan.state == b"snapshot-A"


def test_plan_partitions_after_checkpoint(rig):
    engine, recovery = rig
    # Checkpoint at A, then B (prompt, idempotent) and C (tool, non-idempotent)
    # ran but were never checkpointed — simulating work lost at crash.
    a = node("run-1")
    run(engine.save(a, state=b"snap-A"))
    b = node("run-1", NodeType.PROMPT)
    run(engine.save(b))
    c = node("run-1", NodeType.TOOL)
    run(engine.save(c))

    plan = run(recovery.plan("run-1"))
    assert plan.resume_node_id == a.id
    assert [n.id for n in plan.replay_nodes] == [b.id]
    assert [n.id for n in plan.skipped_nodes] == [c.id]


def test_plan_ignores_nodes_before_checkpoint(rig):
    engine, recovery = rig
    a = node("run-1")
    run(engine.save(a))                      # earlier, no checkpoint
    b = node("run-1")
    run(engine.save(b, state=b"snap-B"))     # checkpoint here
    plan = run(recovery.plan("run-1"))
    assert plan.resume_node_id == b.id
    # a came before the checkpoint node, so it is not a replay candidate.
    assert plan.replay_nodes == []


# --- replay ---------------------------------------------------------------

def test_replay_runs_idempotent_skips_side_effects(rig):
    engine, recovery = rig
    a = node("run-1")
    run(engine.save(a, state=b"snap"))
    b = node("run-1", NodeType.PROMPT, input={"step": "reason"})
    run(engine.save(b))
    c = node("run-1", NodeType.TOOL, input={"charge": 100})
    run(engine.save(c))

    handled = []

    async def handler(n):
        handled.append(n.id)
        return n.id

    plan = run(recovery.plan("run-1"))
    results = run(recovery.replay(plan, handler))

    assert handled == [b.id]                 # tool C never re-executed
    assert results == [b.id]
    assert engine.metrics.counter("recovery.replayed") == 1
    assert engine.metrics.counter("recovery.skipped") == 1


def test_replay_noop_on_fresh_start(rig):
    engine, recovery = rig
    run(engine.save(node("run-1")))
    plan = run(recovery.plan("run-1"))

    async def handler(n):  # pragma: no cover - must never be called
        raise AssertionError("handler called on fresh start")

    assert run(recovery.replay(plan, handler)) == []


def test_recover_convenience(rig):
    engine, recovery = rig
    a = node("run-1")
    run(engine.save(a, state=b"snap"))
    b = node("run-1", NodeType.PROMPT)
    run(engine.save(b))

    seen = []

    async def handler(n):
        seen.append(n.id)

    plan = run(recovery.recover("run-1", handler))
    assert plan.found is True
    assert seen == [b.id]


# --- crash simulation -----------------------------------------------------

class CrashableStore(SQLiteStore):
    """SQLite store that stops persisting after a simulated crash point."""

    def __init__(self):
        super().__init__()
        self.crashed = False

    async def write(self, node):
        if self.crashed:
            return  # writes are silently lost after the crash
        await super().write(node)


def test_crash_after_checkpoint_recovers_to_checkpoint():
    store = CrashableStore()
    try:
        engine = CheckpointEngine(store)
        recovery = RecoveryEngine(engine)

        # Durable progress: checkpoint at node A.
        a = node("run-1")
        run(engine.save(a, state=b"durable-A"))

        # Crash: subsequent writes are lost.
        store.crashed = True
        b = node("run-1", NodeType.PROMPT)
        run(engine.save(b))  # lost in cold store

        # New engine simulating a restarted process with a cold hot-cache.
        fresh_engine = CheckpointEngine(store)
        fresh_recovery = RecoveryEngine(fresh_engine)
        store.crashed = False  # storage back online after restart

        plan = run(fresh_recovery.plan("run-1"))
        assert plan.found is True
        assert plan.resume_node_id == a.id
        assert plan.state == b"durable-A"
        # B never reached the durable log, so nothing to replay.
        assert plan.replay_nodes == []
    finally:
        store.close()
