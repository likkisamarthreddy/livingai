# Replay

Replay is distinct from recovery. **Recovery** is automatic and happens on crash.
**Replay** is manual and happens for debugging, testing, and optimization — it
re-runs a previously recorded execution from the durable log.

```python
from livingai import ReplaySession, ReplayMode

session = ReplaySession(store, "run-1")
results = await session.run(handler, mode=ReplayMode.MOCK_TOOLS)
```

The `handler` is an async callable that performs the (re-)execution of a node and
returns its output. Each `ReplayResult` carries the `node`, its `output`, and a
`mocked` flag.

## Modes

### `FULL`
Re-execute every node from scratch. The handler is called for all nodes.

### `FROM_NODE`
Re-execute from a specific node onward.

```python
await session.run(handler, mode=ReplayMode.FROM_NODE, from_node_id=node_id)
```

Raises `ValueError` if `from_node_id` is missing, `KeyError` if it isn't in the
execution.

### `MOCK_TOOLS`
Re-execute, but **tool calls return their recorded output** instead of hitting
real APIs. This is the most valuable debugging mode: re-run the LLM reasoning
against the exact same tool responses without triggering real side effects.

```python
results = await session.run(handler, mode=ReplayMode.MOCK_TOOLS)
# TOOL nodes -> result.mocked is True, output is the recorded output
# other nodes -> handler is invoked normally
```

### `COUNTERFACTUAL`
Re-execute with a modified input at a specific node to ask "what if…". The stored
log is never mutated — a probe copy of the node is used.

```python
results = await session.run(
    handler,
    mode=ReplayMode.COUNTERFACTUAL,
    counterfactual=(node_id, {"prompt": "changed input"}),
)
```

Raises `ValueError` if `counterfactual` is not supplied.

## Metrics

| Metric | Meaning |
| --- | --- |
| `replay.executed` | Nodes run through the handler |
| `replay.mocked` | Tool nodes served from history (MOCK_TOOLS) |
| `replay.sessions` | Replay runs completed |

See also the [CLI](cli.md) for `livingai replay <execution_id>`.
