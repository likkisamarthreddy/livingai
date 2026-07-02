"""Unit tests for the SQLite checkpoint store.

Tests are written against the :class:`CheckpointStore` protocol so the same
suite can be reused for future backends (Redis, Postgres). They drive the async
API via :func:`asyncio.run`, so no async test plugin is required.
"""

import asyncio

import pytest

from livingai import CheckpointStore, ExecutionNode, NodeType, Status, SQLiteStore


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store():
    s = SQLiteStore()  # in-memory
    yield s
    s.close()


def make_node(execution_id="exec-1", **kwargs):
    return ExecutionNode(execution_id=execution_id, type=NodeType.PROMPT, **kwargs)


def test_satisfies_protocol(store):
    assert isinstance(store, CheckpointStore)


def test_read_missing_returns_none(store):
    assert run(store.read("nope")) is None


def test_write_then_read(store):
    node = make_node(input={"q": "hi"})
    run(store.write(node))
    got = run(store.read(node.id))
    assert got is not None
    assert got.id == node.id
    assert got.input == {"q": "hi"}


def test_write_preserves_checkpoint_blob(store):
    node = make_node(checkpoint=b"\x01\x02\x03")
    run(store.write(node))
    got = run(store.read(node.id))
    assert got.checkpoint == b"\x01\x02\x03"


def test_append_only_read_returns_latest(store):
    node = make_node(status=Status.PENDING)
    run(store.write(node))
    # Same id, advanced state — append, not mutate.
    node.status = Status.SUCCESS
    node.output = {"done": True}
    run(store.write(node))

    got = run(store.read(node.id))
    assert got.status is Status.SUCCESS
    assert got.output == {"done": True}


def test_list_by_execution_orders_by_creation(store):
    a = make_node(execution_id="run-A")
    b = make_node(execution_id="run-A")
    c = make_node(execution_id="run-A")
    for n in (a, b, c):
        run(store.write(n))
    ids = [n.id for n in run(store.list_by_execution("run-A"))]
    assert ids == [a.id, b.id, c.id]


def test_list_by_execution_dedups_to_latest(store):
    a = make_node(execution_id="run-A", status=Status.PENDING)
    run(store.write(a))
    b = make_node(execution_id="run-A", status=Status.PENDING)
    run(store.write(b))
    # Update a again (append).
    a.status = Status.SUCCESS
    run(store.write(a))

    nodes = run(store.list_by_execution("run-A"))
    assert len(nodes) == 2
    by_id = {n.id: n for n in nodes}
    assert by_id[a.id].status is Status.SUCCESS
    # Order preserved by first appearance.
    assert [n.id for n in nodes] == [a.id, b.id]


def test_list_isolates_executions(store):
    run(store.write(make_node(execution_id="run-A")))
    run(store.write(make_node(execution_id="run-B")))
    assert len(run(store.list_by_execution("run-A"))) == 1
    assert len(run(store.list_by_execution("run-B"))) == 1


def test_get_latest_checkpoint_none_when_absent(store):
    run(store.write(make_node(execution_id="run-A")))  # no checkpoint
    assert run(store.get_latest_checkpoint("run-A")) is None


def test_get_latest_checkpoint_returns_most_recent(store):
    first = make_node(execution_id="run-A", checkpoint=b"first")
    run(store.write(first))
    plain = make_node(execution_id="run-A")  # no checkpoint in between
    run(store.write(plain))
    second = make_node(execution_id="run-A", checkpoint=b"second")
    run(store.write(second))

    latest = run(store.get_latest_checkpoint("run-A"))
    assert latest is not None
    assert latest.checkpoint == b"second"


def test_get_latest_checkpoint_isolated_per_execution(store):
    run(store.write(make_node(execution_id="run-A", checkpoint=b"a")))
    run(store.write(make_node(execution_id="run-B", checkpoint=b"b")))
    assert run(store.get_latest_checkpoint("run-A")).checkpoint == b"a"
    assert run(store.get_latest_checkpoint("run-B")).checkpoint == b"b"


def test_persists_to_disk(tmp_path):
    db = str(tmp_path / "livingai.db")
    s1 = SQLiteStore(db)
    node = make_node()
    run(s1.write(node))
    s1.close()

    s2 = SQLiteStore(db)
    try:
        got = run(s2.read(node.id))
        assert got is not None
        assert got.id == node.id
    finally:
        s2.close()


def test_concurrent_writes(store):
    nodes = [make_node(execution_id="run-A") for _ in range(50)]

    async def write_all():
        await asyncio.gather(*(store.write(n) for n in nodes))

    run(write_all())
    stored = run(store.list_by_execution("run-A"))
    assert len(stored) == 50
