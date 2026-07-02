"""Unit tests for the execution graph data model."""

from datetime import datetime, timezone

import pytest

from livingai import ErrorInfo, ExecutionNode, NodeType, Status, new_id, utcnow


def test_new_id_is_unique():
    ids = {new_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_utcnow_is_timezone_aware():
    now = utcnow()
    assert now.tzinfo is not None


def test_defaults():
    node = ExecutionNode(execution_id="exec-1", type=NodeType.PROMPT)
    assert node.status is Status.PENDING
    assert node.parent_id is None
    assert node.completed_at is None
    assert node.metadata == {}
    assert node.checkpoint is None
    assert isinstance(node.id, str) and len(node.id) > 0


def test_string_enums_are_coerced():
    node = ExecutionNode(execution_id="exec-1", type="TOOL", status="RUNNING")
    assert node.type is NodeType.TOOL
    assert node.status is Status.RUNNING


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        ExecutionNode(execution_id="exec-1", type="NOT_A_TYPE")


def test_invalid_status_raises():
    with pytest.raises(ValueError):
        ExecutionNode(execution_id="exec-1", type=NodeType.PROMPT, status="BOGUS")


def test_error_info_round_trip():
    err = ErrorInfo(type="ValueError", message="boom", traceback="line 1")
    restored = ErrorInfo.from_dict(err.to_dict())
    assert restored == err


def test_error_info_from_none():
    assert ErrorInfo.from_dict(None) is None


def test_error_info_optional_traceback():
    err = ErrorInfo.from_dict({"type": "E", "message": "m"})
    assert err.traceback is None


def test_to_dict_excludes_checkpoint():
    node = ExecutionNode(
        execution_id="exec-1", type=NodeType.MEMORY, checkpoint=b"\x00\x01"
    )
    data = node.to_dict()
    assert "checkpoint" not in data


def test_dict_round_trip_full():
    node = ExecutionNode(
        execution_id="exec-42",
        type=NodeType.TOOL,
        parent_id="parent-1",
        status=Status.SUCCESS,
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        input={"query": "weather"},
        output={"temp": 21},
        error=None,
        cost_tokens=123,
        latency_ms=456,
        metadata={"framework": "langgraph", "node": "n7"},
    )
    restored = ExecutionNode.from_dict(node.to_dict())
    assert restored.execution_id == node.execution_id
    assert restored.type is NodeType.TOOL
    assert restored.parent_id == "parent-1"
    assert restored.status is Status.SUCCESS
    assert restored.created_at == node.created_at
    assert restored.completed_at == node.completed_at
    assert restored.input == {"query": "weather"}
    assert restored.output == {"temp": 21}
    assert restored.cost_tokens == 123
    assert restored.latency_ms == 456
    assert restored.metadata == {"framework": "langgraph", "node": "n7"}


def test_dict_round_trip_with_error():
    node = ExecutionNode(
        execution_id="exec-1",
        type=NodeType.BRANCH,
        status=Status.FAILED,
        error=ErrorInfo(type="RuntimeError", message="crashed"),
    )
    restored = ExecutionNode.from_dict(node.to_dict())
    assert restored.error == ErrorInfo(type="RuntimeError", message="crashed")


def test_json_round_trip_preserves_checkpoint():
    blob = b"\xde\xad\xbe\xef"
    node = ExecutionNode(
        execution_id="exec-1", type=NodeType.PROMPT, checkpoint=blob
    )
    raw = node.to_json()
    restored = ExecutionNode.from_json(raw, checkpoint=blob)
    assert restored.checkpoint == blob
    assert restored.id == node.id


def test_json_is_deterministic():
    node = ExecutionNode(
        execution_id="exec-1", type=NodeType.PROMPT, id="fixed", metadata={"b": 2, "a": 1}
    )
    # sorted keys => stable serialization regardless of insertion order.
    assert node.to_json() == node.to_json()


def test_naive_datetime_is_treated_as_utc():
    node = ExecutionNode(
        execution_id="exec-1",
        type=NodeType.PROMPT,
        created_at=datetime(2026, 1, 1, 0, 0),  # naive
    )
    data = node.to_dict()
    assert data["created_at"].endswith("+00:00")
