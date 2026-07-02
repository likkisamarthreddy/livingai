"""Example 02 — Crash simulation and automatic resume.

An agent checkpoints after an expensive step, then the process "crashes" (further
writes are lost). On restart the recovery engine reconstructs state from the last
durable checkpoint and replays only the *idempotent* work — never re-running the
side-effecting tool call.

    python examples/02_crash_recovery.py
"""

import asyncio

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    RecoveryEngine,
    SQLiteStore,
    Status,
)


class CrashableStore(SQLiteStore):
    """A durable store that drops writes after a simulated crash."""

    def __init__(self, path: str):
        super().__init__(path)
        self.crashed = False

    async def write(self, node):
        if self.crashed:
            return  # lost on the floor, exactly like a hard crash mid-write
        await super().write(node)


def node(ntype, **kw):
    return ExecutionNode(execution_id="run-crash", type=ntype, status=Status.SUCCESS, **kw)


async def main() -> None:
    store = CrashableStore("crash-demo.db")
    engine = CheckpointEngine(store)

    # Expensive reasoning step — checkpointed durably.
    reason = node(NodeType.PROMPT, input={"q": "book a flight"}, output="plan ready")
    await engine.save(reason, state=b"agent-state-after-planning")
    print("checkpoint saved after planning step")

    # A tool call with a real side effect (charge a card) runs and is recorded.
    charge = node(NodeType.TOOL, input={"charge_usd": 350}, output={"receipt": "R-123"})
    await engine.save(charge)
    print("tool call executed (card charged)")

    # 💥 CRASH — subsequent writes are lost.
    store.crashed = True
    followup = node(NodeType.PROMPT, input={"q": "confirm"}, output="confirmed")
    await engine.save(followup)  # lost
    print("\n*** process crashed ***\n")

    # --- Restart: brand-new engine with a cold hot-cache, storage back online.
    store.crashed = False
    fresh_engine = CheckpointEngine(store)
    recovery = RecoveryEngine(fresh_engine)

    plan = await recovery.plan("run-crash")
    print(f"recovery found checkpoint : {plan.found}")
    print(f"resume from node          : {plan.resume_node_id[:8]}")
    print(f"restored state            : {plan.state!r}")

    replayed = []

    async def rerun(n):
        replayed.append(n)
        return n.output

    await recovery.replay(plan, rerun)
    print(f"idempotent nodes replayed : {len(replayed)}")
    print(f"side-effect nodes skipped : {len(plan.skipped_nodes)} (card NOT re-charged)")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
    # tidy up the demo db
    import os

    if os.path.exists("crash-demo.db"):
        os.remove("crash-demo.db")
