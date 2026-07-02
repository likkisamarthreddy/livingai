# Recovery

When a process crashes and restarts, the `RecoveryEngine` reconstructs execution
state from the last durable checkpoint and determines how to resume.

## Recovery flow

1. Ask: does `execution_id` have a checkpoint?
2. If yes:
   1. Load the latest checkpoint (hot tier, then cold store).
   2. Deserialize the execution state.
   3. Replay **idempotent** nodes recorded after the checkpoint.
   4. Resume from the crash point.
3. If no: start a fresh execution.

## Two-step API

The engine separates analysis from execution so it is easy to test and reason
about.

### `plan()` — pure analysis

```python
from livingai import RecoveryEngine

recovery = RecoveryEngine(engine)
plan = await recovery.plan("run-1")
```

Returns an immutable `RecoveryPlan`:

| Field | Meaning |
| --- | --- |
| `found` | Whether a durable checkpoint exists. If `False`, start fresh. |
| `checkpoint_node` | The node to resume from. |
| `state` | The decompressed checkpoint bytes. |
| `replay_nodes` | Idempotent nodes after the checkpoint — safe to re-run. |
| `skipped_nodes` | Non-idempotent nodes after the checkpoint — must NOT re-run. |
| `resume_node_id` | Convenience: `checkpoint_node.id`. |

`plan()` never mutates state.

### `replay()` — drive re-execution

```python
async def handler(node):
    # re-execute a single idempotent node
    return await run_node(node)

results = await recovery.replay(plan, handler)
```

The handler is invoked once per idempotent node, in order. Non-idempotent nodes
are **skipped** (counted, never invoked) so their side effects don't happen
twice.

### `recover()` — convenience

```python
plan = await recovery.recover("run-1", handler)   # plan() + replay()
```

## The critical constraint

> Tool calls with external side effects (API writes, emails, payments) are
> non-idempotent and must never be re-executed during recovery.

The engine enforces this by partitioning post-checkpoint nodes using
`ExecutionNode.is_idempotent()`. A framework adapter is responsible for marking
tool calls correctly (see [Adapters](adapters.md)).

## Metrics

| Metric | Meaning |
| --- | --- |
| `recovery.fresh_start` | No checkpoint found |
| `recovery.planned` | A recovery plan was produced |
| `recovery.replayed` | Idempotent nodes re-run |
| `recovery.skipped` | Non-idempotent nodes skipped |
