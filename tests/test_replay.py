"""Tests for the replay engine and the livingai CLI."""

import asyncio

import pytest

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    ReplayMode,
    ReplaySession,
    SQLiteStore,
)
from livingai import cli


def run(coro):
    return asyncio.run(coro)


def seed(store):
    """Seed a 3-node execution: prompt -> tool -> prompt."""
    engine = CheckpointEngine(store)
    n1 = ExecutionNode(execution_id="run-1", type=NodeType.PROMPT, input={"q": "hi"}, output="thought")
    n2 = ExecutionNode(execution_id="run-1", type=NodeType.TOOL, input={"call": "api"}, output={"temp": 21})
    n3 = ExecutionNode(execution_id="run-1", type=NodeType.PROMPT, input={"q": "sum"}, output="Sunny 21C")
    run(engine.save(n1, state=b"snap"))
    run(engine.save(n2))
    run(engine.save(n3))
    return n1, n2, n3


# --- replay engine --------------------------------------------------------

def test_full_replay_runs_every_node():
    store = SQLiteStore()
    try:
        seed(store)
        session = ReplaySession(store, "run-1")
        seen = []

        async def handler(node):
            seen.append(node.id)
            return f"re:{node.type.value}"

        results = run(session.run(handler, mode=ReplayMode.FULL))
        assert len(results) == 3
        assert len(seen) == 3
        assert all(not r.mocked for r in results)
    finally:
        store.close()


def test_mock_tools_returns_recorded_output_for_tools():
    store = SQLiteStore()
    try:
        _, n2, _ = seed(store)
        session = ReplaySession(store, "run-1")
        called = []

        async def handler(node):
            called.append(node.id)
            return "real-call"

        results = run(session.run(handler, mode=ReplayMode.MOCK_TOOLS))
        # Tool node must not invoke the handler.
        assert n2.id not in called
        tool_result = next(r for r in results if r.node.id == n2.id)
        assert tool_result.mocked is True
        assert tool_result.output == {"temp": 21}
        assert session.metrics.counter("replay.mocked") == 1
    finally:
        store.close()


def test_from_node_replays_suffix():
    store = SQLiteStore()
    try:
        _, n2, n3 = seed(store)
        session = ReplaySession(store, "run-1")

        async def handler(node):
            return node.id

        results = run(session.run(handler, mode=ReplayMode.FROM_NODE, from_node_id=n2.id))
        assert [r.node.id for r in results] == [n2.id, n3.id]
    finally:
        store.close()


def test_from_node_requires_id():
    store = SQLiteStore()
    try:
        seed(store)
        session = ReplaySession(store, "run-1")

        async def handler(node):
            return None

        with pytest.raises(ValueError):
            run(session.run(handler, mode=ReplayMode.FROM_NODE))
    finally:
        store.close()


def test_from_node_unknown_id_raises():
    store = SQLiteStore()
    try:
        seed(store)
        session = ReplaySession(store, "run-1")

        async def handler(node):
            return None

        with pytest.raises(KeyError):
            run(session.run(handler, mode=ReplayMode.FROM_NODE, from_node_id="nope"))
    finally:
        store.close()


def test_counterfactual_overrides_input():
    store = SQLiteStore()
    try:
        n1, _, _ = seed(store)
        session = ReplaySession(store, "run-1")
        captured = {}

        async def handler(node):
            captured[node.id] = node.input
            return node.input

        results = run(
            session.run(
                handler,
                mode=ReplayMode.COUNTERFACTUAL,
                counterfactual=(n1.id, {"q": "changed"}),
            )
        )
        assert captured[n1.id] == {"q": "changed"}
        # Original stored node is untouched.
        first = next(r for r in results if r.node.id == n1.id)
        assert first.output == {"q": "changed"}
    finally:
        store.close()


def test_counterfactual_requires_tuple():
    store = SQLiteStore()
    try:
        seed(store)
        session = ReplaySession(store, "run-1")

        async def handler(node):
            return None

        with pytest.raises(ValueError):
            run(session.run(handler, mode=ReplayMode.COUNTERFACTUAL))
    finally:
        store.close()


# --- CLI ------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "log.db")
    store = SQLiteStore(path)
    seed(store)
    store.close()
    return path


def test_cli_list(db):
    out = cli.cmd_list(db)
    assert "run-1" in out


def test_cli_list_empty(tmp_path):
    empty = str(tmp_path / "empty.db")
    assert "no executions" in cli.cmd_list(empty)


def test_cli_show(db):
    out = cli.cmd_show("run-1", db)
    assert "3 node(s)" in out
    assert "TOOL" in out
    assert "side-effect" in out


def test_cli_show_missing(db):
    assert "no nodes" in cli.cmd_show("ghost", db)


def test_cli_replay_mock_tools(db):
    out = cli.cmd_replay("run-1", db, mode=ReplayMode.MOCK_TOOLS)
    assert "MOCK_TOOLS" in out
    assert "mock" in out  # tool node reconstructed from history


def test_cli_replay_empty(tmp_path):
    empty = str(tmp_path / "empty.db")
    assert "nothing to replay" in cli.cmd_replay("x", empty)


def test_cli_main_dispatch(db, capsys):
    rc = cli.main(["show", "run-1", "--db", db])
    assert rc == 0
    assert "3 node(s)" in capsys.readouterr().out


def test_cli_main_list(db, capsys):
    assert cli.main(["list", "--db", db]) == 0
    assert "run-1" in capsys.readouterr().out


def test_cli_main_replay(db, capsys):
    assert cli.main(["replay", "run-1", "--db", db, "--mode", "FULL"]) == 0
    assert "FULL" in capsys.readouterr().out
