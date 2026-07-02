"""Example 03 — MOCK_TOOLS debugging.

Re-run a recorded execution's reasoning against the *exact same* tool responses,
without triggering real API calls. This is the single most useful debugging mode:
you can change prompt-handling logic and see how the model would have behaved on
the recorded tool outputs.

    python examples/03_mock_tools_debugging.py
"""

import asyncio

from livingai import (
    CheckpointEngine,
    ExecutionNode,
    NodeType,
    ReplayMode,
    ReplaySession,
    SQLiteStore,
    Status,
)


def node(ntype, **kw):
    return ExecutionNode(execution_id="run-debug", type=ntype, status=Status.SUCCESS, **kw)


async def main() -> None:
    store = SQLiteStore()
    engine = CheckpointEngine(store)

    # Record an original run: reason -> call weather API -> summarize.
    await engine.save(node(NodeType.PROMPT, input={"q": "weather?"}, output="need to look up"))
    await engine.save(node(NodeType.TOOL, input={"city": "Paris"}, output={"temp_c": 21, "sky": "clear"}))
    await engine.save(node(NodeType.PROMPT, input={"q": "summarize"}, output="It's 21C and clear."))

    # Replay with tools mocked. The real API is never called; the recorded
    # {"temp_c": 21} response is reused, so reasoning is deterministic.
    real_api_calls = 0

    async def handler(n):
        nonlocal real_api_calls
        if n.type is NodeType.TOOL:
            real_api_calls += 1  # would be a real network call
            return {"temp_c": 999}  # (never happens in MOCK_TOOLS)
        # Re-run "reasoning" — here we just echo, but you'd call your LLM logic.
        return f"[replayed] {n.output}"

    session = ReplaySession(store, "run-debug")
    results = await session.run(handler, mode=ReplayMode.MOCK_TOOLS)

    print("Replay in MOCK_TOOLS mode:")
    for r in results:
        tag = "mock" if r.mocked else "run "
        print(f"  {tag} {r.node.type.value:<7} -> {r.output!r}")

    print(f"\nreal API calls during replay: {real_api_calls} (expected 0)")
    print(f"tool responses served from history: {session.metrics.counter('replay.mocked')}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
