# Quickstart

## Install

```bash
cd livingai_runtime
pip install -e .
```

The core has **no runtime dependencies**. `pip install -e .` also registers the
`livingai` command-line tool.

## Record a checkpointed execution

Everything is `async`. The default `SQLiteStore()` is in-memory; pass a path for
durable persistence.

```python
import asyncio
from livingai import CheckpointEngine, ExecutionNode, NodeType, Status, SQLiteStore


async def main():
    store = SQLiteStore("agent.db")           # durable local log
    engine = CheckpointEngine(store)          # 50 ms budget, zlib compression

    # Start a node.
    node = ExecutionNode(
        execution_id="run-1",
        type=NodeType.PROMPT,
        input={"prompt": "What's the weather in Paris?"},
        status=Status.RUNNING,
    )
    await engine.save(node)

    # Finish it, attaching a serialized agent-state checkpoint.
    node.status = Status.SUCCESS
    node.output = {"answer": "Sunny, 21°C"}
    await engine.save(node, state=b"...serialized agent state...")

    # Read it back (decompressed).
    loaded_node, state = await engine.load(node.id)
    print(loaded_node.status, loaded_node.output)

    store.close()


asyncio.run(main())
```

## Recover after a crash

```python
from livingai import RecoveryEngine

recovery = RecoveryEngine(engine)

async def rerun(node):
    ...  # re-execute an idempotent node

plan = await recovery.plan("run-1")
if plan.found:
    await recovery.replay(plan, rerun)   # replays idempotent nodes, skips side effects
else:
    ...  # start fresh
```

## Debug with replay

```bash
livingai show run-1 --db agent.db
livingai replay run-1 --db agent.db --mode MOCK_TOOLS
```

Next: read [Concepts](concepts.md) to understand the execution graph and storage
tiers.
