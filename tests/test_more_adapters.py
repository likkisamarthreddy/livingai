"""Tests for the CrewAI and OpenAI Agents adapters (shared BaseAdapter)."""

import asyncio

import pytest

from livingai import (
    CheckpointEngine,
    CrewAIAdapter,
    NodeType,
    OpenAIAgentsAdapter,
    SQLiteStore,
    Status,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def engine():
    store = SQLiteStore()
    yield CheckpointEngine(store)
    store.close()


# --- CrewAI ---------------------------------------------------------------

def test_crewai_prompt_node(engine):
    ad = CrewAIAdapter(engine, execution_id="crew-1")
    node = run(ad.on_node_start(name="planning_task", input={"topic": "AI"}))
    assert node.type is NodeType.PROMPT
    assert node.metadata["framework"] == "crewai"
    assert node.metadata["crew_node"] == "planning_task"
    assert node.status is Status.RUNNING


def test_crewai_delegate_is_side_effect(engine):
    ad = CrewAIAdapter(engine, execution_id="crew-1")
    node = run(ad.on_node_start(name="delegate_to_writer"))
    assert node.type is NodeType.TOOL
    assert node.is_idempotent() is False


def test_crewai_end_and_error(engine):
    ad = CrewAIAdapter(engine, execution_id="crew-1")
    node = run(ad.on_node_start(name="write_task"))
    run(ad.on_node_end(node, output="draft", cost_tokens=80))
    loaded, _ = run(engine.load(node.id))
    assert loaded.status is Status.SUCCESS
    assert loaded.output == "draft"

    node2 = run(ad.on_node_start(name="review_task"))
    run(ad.on_node_error(node2, RuntimeError("bad")))
    loaded2, _ = run(engine.load(node2.id))
    assert loaded2.status is Status.FAILED
    assert loaded2.error.type == "RuntimeError"


# --- OpenAI Agents --------------------------------------------------------

def test_openai_function_is_side_effect(engine):
    ad = OpenAIAgentsAdapter(engine, execution_id="oa-1")
    node = run(ad.on_node_start(name="get_weather_function"))
    assert node.type is NodeType.TOOL
    assert node.is_idempotent() is False
    assert node.metadata["framework"] == "openai-agents"
    assert node.metadata["agent_node"] == "get_weather_function"


def test_openai_reasoning_is_idempotent(engine):
    ad = OpenAIAgentsAdapter(engine, execution_id="oa-1")
    node = run(ad.on_node_start(name="reasoning_turn"))
    assert node.type is NodeType.PROMPT
    assert node.is_idempotent() is True


def test_openai_explicit_idempotent_override(engine):
    ad = OpenAIAgentsAdapter(engine, execution_id="oa-1")
    node = run(ad.on_node_start(name="readonly_retrieval", idempotent=True))
    assert node.type is NodeType.TOOL
    assert node.is_idempotent() is True


def test_auto_execution_id(engine):
    assert CrewAIAdapter(engine).execution_id
    assert OpenAIAgentsAdapter(engine).execution_id


def test_end_attaches_checkpoint(engine):
    ad = OpenAIAgentsAdapter(engine, execution_id="oa-1")
    node = run(ad.on_node_start(name="reason"))
    run(ad.on_node_end(node, output="done", state=b"snap"))
    _, state = run(engine.load(node.id))
    assert state == b"snap"
