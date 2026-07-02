# Migration & Integration Guides

How to adopt the Living AI runtime alongside your existing agent stack. The
runtime is additive — it records and protects executions; it does not replace
your orchestration framework.

## From LangGraph's built-in checkpointer

LangGraph ships a `checkpointer` (e.g. `MemorySaver`, `SqliteSaver`) that
persists graph state for resumption. The Living AI runtime is complementary and
adds crash-safe recovery **plus** replay/debugging and cost analytics.

| Concern | LangGraph checkpointer | Living AI runtime |
| --- | --- | --- |
| Resume an interrupted graph | ✅ | ✅ (recovery engine) |
| Skip re-running side-effecting tools on resume | manual | ✅ (idempotency-aware) |
| Replay with mocked tools for debugging | ❌ | ✅ (`MOCK_TOOLS`) |
| Counterfactual "what-if" replay | ❌ | ✅ (`COUNTERFACTUAL`) |
| Per-node cost/latency analytics | ❌ | ✅ (append-only log) |
| 50 ms bounded checkpoint overhead | n/a | ✅ |

You can run both. Keep LangGraph's checkpointer for its native state-threading,
and attach the `LangGraphAdapter` to record the execution graph for recovery and
replay:

```python
from livingai import CheckpointEngine, SQLiteStore
from livingai.adapters import LangGraphAdapter

engine = CheckpointEngine(SQLiteStore("agent.db"))
adapter = LangGraphAdapter(engine, execution_id=thread_id)

# In your node wrappers / callbacks:
node = await adapter.on_node_start(name=lg_node_name, input=state)
try:
    result = run_node(...)
    await adapter.on_node_end(node, output=result, state=serialized_state)
except Exception as exc:
    await adapter.on_node_error(node, exc)
    raise
```

Mark read-only tools as replay-safe so recovery can re-run them:

```python
await adapter.on_node_start(name="search_readonly", idempotent=True)
```

## Integrating with existing state management

If you already serialize agent state (pickle, JSON, protobuf, ...), pass those
bytes straight through as the checkpoint blob — the runtime compresses and stores
them and hands them back verbatim on recovery:

```python
blob = my_serializer.dumps(agent_state)          # your existing format
await engine.save(node, state=blob)

# later, after a crash:
loaded, blob = (await engine.load(node.id))
agent_state = my_serializer.loads(blob)
```

The runtime never inspects the blob, so your serialization format is entirely
your choice. Keep state under a few MB for best results within the 50 ms budget;
very large state should be externalized (store a reference in the blob).

## Choosing a storage backend

| Scenario | Backend |
| --- | --- |
| Local dev, tests, single-process | `SQLiteStore(":memory:")` or a file path |
| Durable single-node production | `SQLiteStore("path.db")` |
| Distributed / high-throughput | Redis (hot) + Postgres (cold) — *planned, Phase 5* |

Because everything targets the `CheckpointStore` protocol, switching backends is
a one-line change and requires no application code changes.

## Gradual adoption checklist

1. Add `livingai` and attach an adapter to **record** executions (no behavior
   change).
2. Verify recovery in staging with a crash-injection test (see
   [examples/02_crash_recovery.py](../examples/02_crash_recovery.py)).
3. Use `livingai replay <execution_id> --mode MOCK_TOOLS` to debug incidents.
4. Add cost dashboards from the append-only log (see
   [examples/04_cost_tracking.py](../examples/04_cost_tracking.py)).
5. Move to a production store when you outgrow single-node SQLite.
