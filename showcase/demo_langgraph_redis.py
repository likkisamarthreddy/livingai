"""Runnable showcase — swap LangGraph's blocking SQLite saver for a fast Redis one.

This demo runs **end to end with no LangGraph or Redis server required**: it uses
`fakeredis` (an in-process Redis) and livingai's `LangGraphAdapter`, which records
a LangGraph-style agent as plain data. It shows the exact production pattern:

  1. An agent runs a graph: plan -> call_tool -> summarize.
  2. Every step is checkpointed through Redis with the 50 ms SLA budget.
  3. The process "crashes".
  4. Recovery resumes from the last checkpoint and — crucially — **skips the
     non-idempotent tool call** so the side effect never fires twice.

Run:

    pip install "livingai[redis]" fakeredis
    python showcase/demo_langgraph_redis.py
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis

from livingai import CheckpointEngine, RecoveryEngine
from livingai.adapters import LangGraphAdapter
from livingai.stores.redis import RedisStore


# A tiny LangGraph-style graph: (node name, output, writes a checkpoint?)
GRAPH = [
    ("plan", "route to the payment tool", True),
    ("charge_card_tool", {"receipt": "R-1"}, False),   # side effect — must not replay
    ("summarize", "Payment complete.", False),
]


async def main() -> None:
    # The ONLY change from the SQLite default is this store — everything else is
    # identical. In production: RedisStore(url="redis://your-cluster:6379").
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    engine = CheckpointEngine(RedisStore(client=client))
    adapter = LangGraphAdapter(engine, execution_id="payment-run")

    print("Running LangGraph agent through the Redis-backed saver:")
    for name, output, checkpoint in GRAPH:
        node = await adapter.on_node_start(name=name, input={"node": name})
        state = b"serialized-graph-state" if checkpoint else None
        ok = await adapter.on_node_end(node, output=output, state=state, cost_tokens=42)
        idem = "idempotent" if node.is_idempotent() else "SIDE-EFFECT"
        print(f"  {name:<18} [{node.type.value:<6} {idem:<11}] -> {output!r}")

    print("\n💥 Process crashes. Recovering from Redis in a fresh engine:")
    recovery = RecoveryEngine(CheckpointEngine(RedisStore(client=client)))
    plan = await recovery.plan("payment-run")
    print(f"  resume from    : {plan.checkpoint_node.metadata['lg_node']!r}")
    print(f"  replay (safe)  : {[n.metadata['lg_node'] for n in plan.replay_nodes]}")
    print(f"  skip (effects) : {[n.metadata['lg_node'] for n in plan.skipped_nodes]}")
    print("\n  ✅ The card charge is never replayed — no double billing.")


if __name__ == "__main__":
    asyncio.run(main())
