"""Example 05 — Instrumenting a LangGraph-style agent.

Uses ``LangGraphAdapter`` to record an agent's node lifecycle, then recovers it
after a crash. The adapter needs **no** langgraph install — it consumes events as
plain data — so this runs anywhere. Swap the ``run_node`` body for real LangGraph
callbacks in production.

    python examples/05_langgraph_agent.py
"""

import asyncio

from livingai import CheckpointEngine, RecoveryEngine, SQLiteStore
from livingai.adapters import LangGraphAdapter


# A tiny simulated LangGraph: node name -> (output, produces_checkpoint).
GRAPH = [
    ("plan", "route to weather tool", True),
    ("call_weather_api", {"temp_c": 21}, False),   # tool -> non-idempotent
    ("summarize", "It's 21C in Paris.", False),
]


async def main() -> None:
    store = SQLiteStore("lg-demo.db")
    engine = CheckpointEngine(store)
    adapter = LangGraphAdapter(engine, execution_id="lg-run")

    print("Running agent:")
    for name, output, checkpoint in GRAPH:
        node = await adapter.on_node_start(name=name, input={"node": name})
        state = b"agent-state-snapshot" if checkpoint else None
        await adapter.on_node_end(node, output=output, state=state, cost_tokens=50)
        kind = node.type.value
        idem = "idempotent" if node.is_idempotent() else "side-effect"
        print(f"  {name:<18} [{kind}, {idem}] -> {output!r}")

    # Now simulate recovering this run in a fresh process.
    print("\nRecovering after a hypothetical crash:")
    recovery = RecoveryEngine(CheckpointEngine(store))
    plan = await recovery.plan("lg-run")
    print(f"  resume from     : {plan.checkpoint_node.metadata['lg_node']}")
    print(f"  replay (safe)   : {[n.metadata['lg_node'] for n in plan.replay_nodes]}")
    print(f"  skip (effects)  : {[n.metadata['lg_node'] for n in plan.skipped_nodes]}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
    import os

    if os.path.exists("lg-demo.db"):
        os.remove("lg-demo.db")
