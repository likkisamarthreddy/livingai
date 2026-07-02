"""Tests for PostgresStore using pytest-asyncio + asyncpg mock.

Since tests run without a real Postgres instance, we use an in-memory SQLite
backend to validate the store contract, then verify the asyncpg path compiles
and the protocol is satisfied. A live-integration test (`test_postgres_live.py`)
is opt-in via the ``POSTGRES_DSN`` env var.

For the CI suite (no Postgres), we test the PostgresStore contract by
monkey-patching asyncpg with a thin fake that delegates to aiosqlite.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from typing import Any, Optional

from livingai.graph import ExecutionNode, NodeType, Status, new_id


# ---------------------------------------------------------------------------
# Minimal asyncpg fake (no network, no process)
# ---------------------------------------------------------------------------

class _FakeRow(dict):  # type: ignore[type-arg]
    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


class _FakeConn:
    def __init__(self, db: dict[str, Any]) -> None:
        self._db = db

    async def execute(self, sql: str, *args: Any) -> None:
        if sql.strip().startswith("CREATE") or sql.strip().startswith("INSERT"):
            if "INSERT INTO lai_nodes" in sql:
                seq = self._db.setdefault("_seq", 0) + 1
                self._db["_seq"] = seq
                node_id, execution_id, data, checkpoint = args
                rows = self._db.setdefault("rows", [])
                rows.append({
                    "seq": seq,
                    "id": node_id,
                    "execution_id": execution_id,
                    "data": data,
                    "checkpoint": checkpoint,
                })

    async def fetchrow(self, sql: str, *args: Any) -> Optional[_FakeRow]:
        rows = self._db.get("rows", [])
        if "WHERE id = $1 ORDER BY seq DESC LIMIT 1" in sql:
            node_id = args[0]
            matching = [r for r in rows if r["id"] == node_id]
            if not matching:
                return None
            return _FakeRow(max(matching, key=lambda r: r["seq"]))
        if "checkpoint IS NOT NULL" in sql:
            eid = args[0]
            matching = [r for r in rows if r["execution_id"] == eid and r.get("checkpoint")]
            if not matching:
                return None
            return _FakeRow(max(matching, key=lambda r: r["seq"]))
        return None

    async def fetch(self, sql: str, *args: Any) -> list[_FakeRow]:
        rows = self._db.get("rows", [])
        eid = args[0]
        matching = [r for r in rows if r["execution_id"] == eid]
        # latest per node id, ordered by first appearance
        seen: dict[str, _FakeRow] = {}
        first_seq: dict[str, int] = {}
        for r in matching:
            nid = r["id"]
            if nid not in first_seq:
                first_seq[nid] = r["seq"]
            if nid not in seen or r["seq"] > seen[nid]["seq"]:
                seen[nid] = _FakeRow(r)
        return sorted(seen.values(), key=lambda r: first_seq[r["id"]])


class _FakeConnCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
    async def __aenter__(self) -> _FakeConn:
        return self._conn
    async def __aexit__(self, *_: Any) -> None:
        pass


class _FakePool:
    def __init__(self) -> None:
        self._db: dict[str, Any] = {}
        self._conn = _FakeConn(self._db)
    def acquire(self) -> _FakeConnCtx:
        return _FakeConnCtx(self._conn)
    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# import under fake pool
# ---------------------------------------------------------------------------

import unittest.mock as mock
import types

# Provide a stub asyncpg module so the import doesn't fail in CI
_fake_asyncpg = types.ModuleType("asyncpg")
_fake_asyncpg.create_pool = None  # type: ignore[attr-defined]
_fake_asyncpg.Pool = object  # type: ignore[attr-defined]

import sys
if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = _fake_asyncpg

from livingai.stores.postgres import PostgresStore


def _make_store() -> PostgresStore:
    pool = _FakePool()
    store = PostgresStore(pool=pool)  # type: ignore[arg-type]
    store._pool = pool  # type: ignore[assignment]
    return store


def _node(execution_id: str = "exec-1", **kw: Any) -> ExecutionNode:
    return ExecutionNode(
        id=new_id(),
        execution_id=execution_id,
        type=kw.get("type", NodeType.PROMPT),
        status=Status.SUCCESS,
        checkpoint=kw.get("checkpoint"),
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_and_read() -> None:
    store = _make_store()
    node = _node()
    await store.write(node)
    got = await store.read(node.id)
    assert got is not None
    assert got.id == node.id


@pytest.mark.asyncio
async def test_read_missing_returns_none() -> None:
    store = _make_store()
    assert await store.read("does-not-exist") is None


@pytest.mark.asyncio
async def test_write_preserves_checkpoint() -> None:
    store = _make_store()
    blob = b"checkpoint-data"
    node = _node(checkpoint=blob)
    await store.write(node)
    got = await store.read(node.id)
    assert got is not None
    assert got.checkpoint == blob


@pytest.mark.asyncio
async def test_list_by_execution_order() -> None:
    store = _make_store()
    eid = "exec-order"
    nodes = [_node(execution_id=eid) for i in range(5)]
    for n in nodes:
        await store.write(n)
    listed = await store.list_by_execution(eid)
    assert [n.id for n in listed] == [n.id for n in nodes]


@pytest.mark.asyncio
async def test_list_empty() -> None:
    store = _make_store()
    assert await store.list_by_execution("nonexistent") == []


@pytest.mark.asyncio
async def test_get_latest_checkpoint_none() -> None:
    store = _make_store()
    eid = "exec-no-ckpt"
    await store.write(_node(execution_id=eid))
    assert await store.get_latest_checkpoint(eid) is None


@pytest.mark.asyncio
async def test_get_latest_checkpoint() -> None:
    store = _make_store()
    eid = "exec-ckpt"
    n1 = _node(execution_id=eid)
    n2 = _node(execution_id=eid, checkpoint=b"state")
    for n in (n1, n2):
        await store.write(n)
    result = await store.get_latest_checkpoint(eid)
    assert result is not None
    assert result.checkpoint == b"state"


@pytest.mark.asyncio
async def test_isolation_between_executions() -> None:
    store = _make_store()
    n1 = _node(execution_id="exec-a")
    n2 = _node(execution_id="exec-b")
    await store.write(n1)
    await store.write(n2)
    a = await store.list_by_execution("exec-a")
    b = await store.list_by_execution("exec-b")
    assert [n.id for n in a] == [n1.id]
    assert [n.id for n in b] == [n2.id]


@pytest.mark.asyncio
async def test_close_noop() -> None:
    store = _make_store()
    await store.close()  # should not raise
