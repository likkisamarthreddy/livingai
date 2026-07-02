"""Tests for the LangGraph adapter."""

import asyncio

import pytest

from livingai import (
    CheckpointEngine,
    NodeType,
    RecoveryEngine,
    SQLiteStore,
    Status,
)
from livingai.adapters import LangGraphAdapter
from livingai.adapters.langgraph import classify


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def adapter():
    store = SQLiteStore()
    engine = CheckpointEngine(store)
    yield LangGraphAdapter(engine, execution_id="run-1"), engine
    store.close()


# --- classification -------------------------------------------------------

def test_classify_tool_by_name():
    assert classify("call_search_api") is NodeType.TOOL
    assert classify("write_to_db") is NodeType.TOOL


def test_classify_prompt_default():
    assert classify("reasoning_step") is NodeType.PROMPT


def test_classify_explicit_type_wins():
    assert classify("call_api", node_type=NodeType.MEMORY) is NodeType.MEMORY


# --- lifecycle ------------------------------------------------------------

def test_on_node_start_persists_running(adapter):
    ad, engine = adapter
    node = run(ad.on_node_start(name="reason", input={"q": "hi"}))
    assert node.status is Status.RUNNING
    assert node.metadata["framework"] == "langgraph"
    assert node.metadata["lg_node"] == "reason"
    loaded, _ = run(engine.load(node.id))
    assert loaded.status is Status.RUNNING


def test_on_node_end_marks_success_with_checkpoint(adapter):
    ad, engine = adapter
    node = run(ad.on_node_start(name="reason"))
    ok = run(ad.on_node_end(node, output={"answer": 42}, state=b"state", cost_tokens=10))
    assert ok is True
    loaded, state = run(engine.load(node.id))
    assert loaded.status is Status.SUCCESS
    assert loaded.output == {"answer": 42}
    assert loaded.cost_tokens == 10
    assert loaded.latency_ms is not None
    assert state == b"state"


def test_on_node_error_records_error(adapter):
    ad, engine = adapter
    node = run(ad.on_node_start(name="reason"))
    run(ad.on_node_error(node, ValueError("boom")))
    loaded, _ = run(engine.load(node.id))
    assert loaded.status is Status.FAILED
    assert loaded.error.type == "ValueError"
    assert loaded.error.message == "boom"


def test_tool_node_is_non_idempotent(adapter):
    ad, _ = adapter
    node = run(ad.on_node_start(name="call_payment_api"))
    assert node.type is NodeType.TOOL
    assert node.is_idempotent() is False


def test_explicit_idempotent_flag(adapter):
    ad, _ = adapter
    node = run(ad.on_node_start(name="call_readonly_api", idempotent=True))
    assert node.type is NodeType.TOOL
    assert node.is_idempotent() is True


def test_custom_metadata_merged(adapter):
    ad, _ = adapter
    node = run(ad.on_node_start(name="reason", metadata={"thread": "t1"}))
    assert node.metadata["thread"] == "t1"
    assert node.metadata["framework"] == "langgraph"


def test_auto_execution_id():
    store = SQLiteStore()
    try:
        ad = LangGraphAdapter(CheckpointEngine(store))
        assert isinstance(ad.execution_id, str) and ad.execution_id
    finally:
        store.close()


# --- end-to-end: adapter feeds recovery -----------------------------------

def test_adapter_run_then_recover():
    store = SQLiteStore()
    try:
        engine = CheckpointEngine(store)
        ad = LangGraphAdapter(engine, execution_id="run-1")
        recovery = RecoveryEngine(engine)

        # Step 1: a prompt node with a checkpoint.
        n1 = run(ad.on_node_start(name="reason", input={"q": "weather"}))
        run(ad.on_node_end(n1, output="thinking", state=b"snapshot"))

        # Step 2: a tool node (side effect) that ran after the checkpoint.
        n2 = run(ad.on_node_start(name="call_weather_api"))
        run(ad.on_node_end(n2, output={"temp": 21}))

        # Step 3: another prompt node (idempotent) after the checkpoint.
        n3 = run(ad.on_node_start(name="summarize"))
        run(ad.on_node_end(n3, output="Sunny 21C"))

        plan = run(recovery.plan("run-1"))
        assert plan.resume_node_id == n1.id
        replay_ids = {n.id for n in plan.replay_nodes}
        skipped_ids = {n.id for n in plan.skipped_nodes}
        assert n3.id in replay_ids
        assert n2.id in skipped_ids  # tool call never re-run
    finally:
        store.close()
