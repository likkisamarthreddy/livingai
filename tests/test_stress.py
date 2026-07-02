"""Stress tests — large graphs, concurrent writers, write contention.

These are heavier than the unit tests but still fast enough for CI. They assert
correctness under load, not wall-clock performance.
"""

import asyncio

import pytest

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    RecoveryEngine,
    ReplayMode,
    ReplaySession,
    SQLiteStore,
    Status,
)


def run(coro):
    return asyncio.run(coro)


def make_node(execution_id, ntype=NodeType.PROMPT, status=Status.SUCCESS, **kw):
    return ExecutionNode(execution_id=execution_id, type=ntype, status=status, **kw)


def test_large_execution_graph(tmp_path):
    """10k nodes persist and read back with correct ordering and dedup."""
    store = SQLiteStore(str(tmp_path / "big.db"))
    try:
        n = 10_000

        # Write straight to the durable store: this test measures store capacity
        # and ordering, not the CheckpointEngine's 50 ms overhead budget (which
        # deliberately drops writes that exceed it under load).
        async def write_all():
            for i in range(n):
                await store.write(make_node("big", input={"i": i}))

        run(write_all())
        nodes = run(store.list_by_execution("big"))
        assert len(nodes) == n
        # Order preserved by first appearance.
        assert [node.input["i"] for node in nodes[:5]] == [0, 1, 2, 3, 4]
        assert nodes[-1].input["i"] == n - 1
    finally:
        store.close()


def test_concurrent_writers_no_loss():
    """Many coroutines writing concurrently: every write is durably recorded."""
    store = SQLiteStore()
    try:
        engine = CheckpointEngine(store, overhead_budget_ms=60_000)
        total = 500

        async def worker(worker_id):
            for j in range(50):
                await engine.save(make_node("conc", metadata={"w": worker_id, "j": j}))

        async def run_all():
            await asyncio.gather(*(worker(w) for w in range(10)))

        run(run_all())
        nodes = run(store.list_by_execution("conc"))
        assert len(nodes) == total
    finally:
        store.close()


def test_write_contention_updates_same_nodes():
    """Repeatedly appending new states of the same nodes keeps latest projection."""
    store = SQLiteStore()
    try:
        engine = CheckpointEngine(store, overhead_budget_ms=60_000)
        nodes = [make_node("cont", status=Status.PENDING) for _ in range(100)]

        async def churn():
            for _ in range(5):  # 5 status transitions each -> 500 appends
                for node in nodes:
                    node.status = Status.RUNNING if node.status is Status.PENDING else Status.SUCCESS
                    await engine.save(node)

        run(churn())
        latest = run(store.list_by_execution("cont"))
        assert len(latest) == 100  # deduped to latest projection
        assert all(node.status is Status.SUCCESS for node in latest)
    finally:
        store.close()


def test_recovery_scales_with_large_log():
    """Recovery planning over a large post-checkpoint tail partitions correctly."""
    store = SQLiteStore()
    try:
        engine = CheckpointEngine(store, overhead_budget_ms=60_000)
        recovery = RecoveryEngine(engine)

        run(engine.save(make_node("rec"), state=b"snap"))
        # 1000 idempotent + 1000 side-effecting nodes after the checkpoint.
        async def tail():
            for i in range(1000):
                await engine.save(make_node("rec", NodeType.PROMPT))
                await engine.save(make_node("rec", NodeType.TOOL))

        run(tail())
        plan = run(recovery.plan("rec"))
        assert len(plan.replay_nodes) == 1000
        assert len(plan.skipped_nodes) == 1000
    finally:
        store.close()


def test_replay_large_graph():
    store = SQLiteStore()
    try:
        engine = CheckpointEngine(store, overhead_budget_ms=60_000)

        async def seed():
            for i in range(2000):
                await engine.save(make_node("rep", output=i))

        run(seed())
        session = ReplaySession(store, "rep")

        async def handler(node):
            return node.output

        results = run(session.run(handler, mode=ReplayMode.FULL))
        assert len(results) == 2000
    finally:
        store.close()

