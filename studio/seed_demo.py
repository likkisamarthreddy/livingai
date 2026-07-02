"""Seed a demo database so you can explore Living AI Studio immediately.

Creates `studio_demo.db` with a few executions: a clean run, a run with a failed
node, and a multi-tool payment run — then you can launch the visualizer.

Run:

    python studio/seed_demo.py
    streamlit run studio/app.py -- --db studio_demo.db
"""

from __future__ import annotations

import asyncio

from livingai import CheckpointEngine, ExecutionNode, NodeType, SQLiteStore, Status

DB = "studio_demo.db"

RUNS = {
    "weather-run": [
        ("plan", NodeType.PROMPT, Status.SUCCESS, "route to weather tool", True),
        ("get_weather", NodeType.TOOL, Status.SUCCESS, {"temp_c": 21}, False),
        ("summarize", NodeType.PROMPT, Status.SUCCESS, "It's 21C in Paris.", False),
    ],
    "payment-run": [
        ("plan", NodeType.PROMPT, Status.SUCCESS, "route to payment", True),
        ("charge_card", NodeType.TOOL, Status.SUCCESS, {"receipt": "R-1"}, False),
        ("email_receipt", NodeType.TOOL, Status.SUCCESS, {"sent": True}, False),
        ("summarize", NodeType.PROMPT, Status.SUCCESS, "Payment complete.", False),
    ],
    "failing-run": [
        ("plan", NodeType.PROMPT, Status.SUCCESS, "route to search", True),
        ("search_api", NodeType.TOOL, Status.FAILED, "HTTP 503", False),
    ],
}


async def main() -> None:
    store = SQLiteStore(DB)
    engine = CheckpointEngine(store)
    for execution_id, steps in RUNS.items():
        for name, ntype, status, output, checkpoint in steps:
            node = ExecutionNode(
                execution_id=execution_id,
                type=ntype,
                status=status,
                output=output,
                cost_tokens=42,
                metadata={"name": name},
            )
            state = b"serialized-state" if checkpoint else None
            await engine.save(node, state=state)
    store.close()
    print(f"Seeded {len(RUNS)} executions into {DB!r}.")
    print("Now run:  streamlit run studio/app.py -- --db studio_demo.db")


if __name__ == "__main__":
    asyncio.run(main())
