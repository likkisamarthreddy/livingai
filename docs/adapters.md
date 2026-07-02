# Adapters

Adapters are **thin translation layers** that map a framework's events onto the
runtime's `ExecutionNode` model. Per the framework-agnostic principle, the core
never imports a framework, and adapters never leak framework types into the core.

## LangGraph

`LangGraphAdapter` records a LangGraph run as execution nodes. It does **not**
import `langgraph` — events are consumed as plain data, so it runs anywhere and
keeps the core zero-dependency.

```python
from livingai import CheckpointEngine, SQLiteStore
from livingai.adapters import LangGraphAdapter

engine = CheckpointEngine(SQLiteStore("agent.db"))
adapter = LangGraphAdapter(engine, execution_id="run-1")

# Node starts
node = await adapter.on_node_start(name="retrieve_docs", input={"q": "weather"})

# Node succeeds (optionally attach a checkpoint blob)
await adapter.on_node_end(node, output=docs, state=serialized_state, cost_tokens=42)

# ...or fails
# await adapter.on_node_error(node, exc)
```

### Node classification

Node names are mapped to a `NodeType` heuristically: names containing tool-like
hints (`tool`, `action`, `call`, `api`, `http`, `search`, `fetch`, `write`)
become `TOOL` nodes; everything else defaults to `PROMPT`. Tool nodes are marked
**non-idempotent** so recovery won't re-trigger their side effects.

Override per node when you know better:

```python
# A read-only API is safe to replay:
await adapter.on_node_start(name="call_readonly_api", idempotent=True)

# Force a specific type:
from livingai import NodeType
await adapter.on_node_start(name="lookup", node_type=NodeType.MEMORY)
```

### Lifecycle summary

| Method | Effect |
| --- | --- |
| `on_node_start(name, ...)` | Creates and persists a `RUNNING` node. |
| `on_node_end(node, output=, state=, cost_tokens=)` | Marks `SUCCESS`, sets latency, attaches checkpoint. |
| `on_node_error(node, exc)` | Marks `FAILED` with structured `ErrorInfo`. |

Recorded runs flow straight into the [Recovery](recovery.md) and
[Replay](replay.md) engines.

## Writing your own adapter

Implement the same pattern for any framework by subclassing
`livingai.adapters._base.BaseAdapter` and setting three attributes:

```python
from livingai.adapters._base import BaseAdapter

class MyFrameworkAdapter(BaseAdapter):
    framework = "myframework"          # metadata["framework"] tag
    node_key = "mf_node"               # metadata key for the node name
    tool_hints = ("tool", "call")      # names that mark side-effecting TOOL nodes
```

The built-in `CrewAIAdapter` and `OpenAIAgentsAdapter` are ~10 lines each on top
of `BaseAdapter` — use them as templates. On each event, the base constructs an
`ExecutionNode`, sets `type`/`status`/`metadata` (including `idempotent` for
side-effecting steps), and persists it with `CheckpointEngine.save`.
