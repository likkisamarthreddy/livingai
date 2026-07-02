"""Example 01 — Basic checkpointing.

Record a two-step execution, attach a serialized state checkpoint, and read it
back. Run with:

    python examples/01_basic_checkpoint.py
"""

import asyncio

from livingai import CheckpointEngine, ExecutionNode, NodeType, Status, SQLiteStore


async def main() -> None:
    store = SQLiteStore()  # in-memory; pass a path for durability
    engine = CheckpointEngine(store)

    # Step 1: a prompt node begins.
    node = ExecutionNode(
        execution_id="demo-run",
        type=NodeType.PROMPT,
        input={"prompt": "What's the weather in Paris?"},
        status=Status.RUNNING,
    )
    await engine.save(node)
    print(f"started  {node.type.value} {node.id[:8]} — {node.status.value}")

    # Step 2: it completes; attach the (serialized) agent state as a checkpoint.
    node.status = Status.SUCCESS
    node.output = {"answer": "Sunny, 21C"}
    ok = await engine.save(node, state=b"...serialized agent state...")
    print(f"saved    checkpoint within budget: {ok}")

    # Read it back — the state is transparently decompressed.
    loaded, state = await engine.load(node.id)
    print(f"loaded   {loaded.status.value} output={loaded.output} state={state!r}")

    p99 = engine.metrics.percentile("checkpoint.write_ms", 99)
    print(f"metrics  writes={engine.metrics.counter('checkpoint.write')} p99={p99:.3f}ms")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
