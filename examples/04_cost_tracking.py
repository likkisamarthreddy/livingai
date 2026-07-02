"""Example 04 — Cost tracking across retries.

Every node can record ``cost_tokens``. Because the log is append-only, retries of
a failing step are all preserved, so you can aggregate the *true* cost of an
execution — including the wasted tokens spent on attempts that failed.

    python examples/04_cost_tracking.py
"""

import asyncio

from livingai import (
    CheckpointEngine,
    ErrorInfo,
    ExecutionNode,
    NodeType,
    SQLiteStore,
    Status,
)


async def main() -> None:
    store = SQLiteStore()
    engine = CheckpointEngine(store)

    # A flaky step: two failed attempts (still cost tokens) then success.
    attempts = [
        (Status.FAILED, 120, "rate limited"),
        (Status.FAILED, 118, "timeout"),
        (Status.SUCCESS, 130, None),
    ]
    for i, (status, tokens, err) in enumerate(attempts, start=1):
        n = ExecutionNode(
            execution_id="run-cost",
            type=NodeType.PROMPT,
            input={"attempt": i},
            status=status,
            cost_tokens=tokens,
            error=ErrorInfo(type="RetryableError", message=err) if err else None,
        )
        await engine.save(n)

    # A tool call and a final summary.
    await engine.save(ExecutionNode(execution_id="run-cost", type=NodeType.TOOL,
                                    status=Status.SUCCESS, cost_tokens=0))
    await engine.save(ExecutionNode(execution_id="run-cost", type=NodeType.PROMPT,
                                    status=Status.SUCCESS, cost_tokens=64))

    # Aggregate from the durable log.
    nodes = await store.list_by_execution("run-cost")
    total = sum(n.cost_tokens or 0 for n in nodes)
    wasted = sum(n.cost_tokens or 0 for n in nodes if n.status is Status.FAILED)

    print(f"nodes recorded      : {len(nodes)}")
    print(f"total tokens        : {total}")
    print(f"wasted on retries   : {wasted} ({wasted / total:.0%} of spend)")
    print(f"failed attempts     : {sum(1 for n in nodes if n.status is Status.FAILED)}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
