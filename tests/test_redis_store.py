"""Tests for RedisStore using fakeredis (no running Redis needed).

    pip install fakeredis
"""
from __future__ import annotations

import asyncio
import pytest

pytest.importorskip("fakeredis", reason="fakeredis not installed")
pytest.importorskip("redis", reason="redis package not installed")

import fakeredis.aioredis as fakeredis_async  # type: ignore[import-untyped]

from livingai.stores.redis import RedisStore
from livingai.graph import ExecutionNode, NodeType, Status, new_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node(execution_id: str = "exec-1", **kw: object) -> ExecutionNode:
    return ExecutionNode(
        id=new_id(),
        execution_id=execution_id,
        type=kw.get("type", NodeType.PROMPT),  # type: ignore[arg-type]
        status=Status.SUCCESS,
        checkpoint=kw.get("checkpoint"),  # type: ignore[arg-type]
    )


def make_store() -> RedisStore:
    fake = fakeredis_async.FakeRedis(decode_responses=False)
    return RedisStore(client=fake)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_and_read() -> None:
    store = make_store()
    node = _node()
    await store.write(node)
    got = await store.read(node.id)
    assert got is not None
    assert got.id == node.id
    assert got.execution_id == node.execution_id


@pytest.mark.asyncio
async def test_read_missing_returns_none() -> None:
    store = make_store()
    assert await store.read("does-not-exist") is None


@pytest.mark.asyncio
async def test_write_preserves_checkpoint_blob() -> None:
    store = make_store()
    blob = b"\x01\x02\x03checkpoint"
    node = _node(checkpoint=blob)
    await store.write(node)
    got = await store.read(node.id)
    assert got is not None
    assert got.checkpoint == blob


@pytest.mark.asyncio
async def test_list_by_execution_order() -> None:
    store = make_store()
    eid = "exec-order"
    nodes = [_node(execution_id=eid) for i in range(5)]
    for n in nodes:
        await store.write(n)
    listed = await store.list_by_execution(eid)
    assert [n.id for n in listed] == [n.id for n in nodes]


@pytest.mark.asyncio
async def test_list_by_execution_empty() -> None:
    store = make_store()
    assert await store.list_by_execution("nonexistent") == []


@pytest.mark.asyncio
async def test_list_latest_projection() -> None:
    """Overwriting a node updates the latest projection."""
    store = make_store()
    eid = "exec-proj"
    node = _node(execution_id=eid)
    await store.write(node)
    updated = ExecutionNode(
        id=node.id,
        execution_id=eid,
        type=NodeType.TOOL,
        status=Status.SUCCESS,
    )
    await store.write(updated)
    got = await store.read(node.id)
    assert got is not None
    assert got.type == NodeType.TOOL
    # list_by_execution returns only one entry per node
    listed = await store.list_by_execution(eid)
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_get_latest_checkpoint_none() -> None:
    store = make_store()
    eid = "exec-no-ckpt"
    await store.write(_node(execution_id=eid))
    assert await store.get_latest_checkpoint(eid) is None


@pytest.mark.asyncio
async def test_get_latest_checkpoint_returns_last() -> None:
    store = make_store()
    eid = "exec-ckpt"
    n1 = _node(execution_id=eid)
    n2 = _node(execution_id=eid, checkpoint=b"state-data")
    n3 = _node(execution_id=eid)
    for n in (n1, n2, n3):
        await store.write(n)
    result = await store.get_latest_checkpoint(eid)
    assert result is not None
    assert result.checkpoint == b"state-data"


@pytest.mark.asyncio
async def test_isolation_between_executions() -> None:
    store = make_store()
    n1 = _node(execution_id="exec-a")
    n2 = _node(execution_id="exec-b")
    await store.write(n1)
    await store.write(n2)
    assert [n.id for n in await store.list_by_execution("exec-a")] == [n1.id]
    assert [n.id for n in await store.list_by_execution("exec-b")] == [n2.id]


@pytest.mark.asyncio
async def test_close() -> None:
    store = make_store()
    await store.close()  # should not raise
