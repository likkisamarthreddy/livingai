# Reddit posts

## r/Python  (highest traffic, most likely to get stars)

**Title:**
I built a crash recovery and replay library for AI agents — zero dependencies, works with LangGraph/CrewAI/OpenAI

**Body:**
Been working on something I wish existed when I started building agents.

The problem: agents crash after expensive work. A process dies mid-workflow after the LLM called, after the tool charged the card, after three steps of a ten-step plan. Everything's gone. You restart from zero, repay for the tokens, and hope the tool side effects don't fire twice.

**livingai** records every step to an append-only log and handles the recovery:

```python
pip install livingai
```

```python
import asyncio
from livingai import CheckpointEngine, ExecutionNode, NodeType, RecoveryEngine, SQLiteStore, Status

async def main():
    engine = CheckpointEngine(SQLiteStore("agent.db"))

    # After each step, save it
    step = ExecutionNode(execution_id="run-1", type=NodeType.PROMPT,
                         status=Status.SUCCESS, output="plan ready")
    await engine.save(step, state=b"...serialized state...")

    # Tool with side effects (charges a card)
    charge = ExecutionNode(execution_id="run-1", type=NodeType.TOOL,
                           status=Status.SUCCESS, output={"receipt": "R-1"})
    await engine.save(charge)

    # 💥 Process crashes. On restart:
    recovery = RecoveryEngine(CheckpointEngine(SQLiteStore("agent.db")))
    plan = await recovery.plan("run-1")
    print("replay safe :", len(plan.replay_nodes))     # idempotent work to redo
    print("skip effects:", len(plan.skipped_nodes))    # card is NOT re-charged

asyncio.run(main())
```

Output:
```
replay safe : 0
skip effects: 1    # the card charge is never replayed
```

**What else it does:**
- `MOCK_TOOLS` replay: re-run a recorded execution returning stored tool responses — debug reasoning without real API calls or charges
- `COUNTERFACTUAL` mode: what would have happened with different inputs
- Three framework adapters: LangGraph, CrewAI, OpenAI Agents SDK
- Optional Redis/PostgreSQL backends: `pip install livingai[redis]`
- CLI: `livingai list/show/replay --db agent.db`
- 128 tests, 100% coverage, mypy strict, CI on Python 3.9–3.12

GitHub: https://github.com/likkisamarthreddy/livingai

---

## r/LangChain

**Title:**
LangGraph crash recovery without re-running tool side effects — new open-source library

**Body:**
Built a recovery layer specifically for agents: it records every node to an append-only log, and on crash it knows which nodes are safe to replay (LLM calls) vs which must be skipped (tool calls with side effects like payments/emails).

The LangGraph adapter is a thin wrapper — no framework imports in the core, just metadata tagging:

```python
from livingai.adapters import LangGraphAdapter
from livingai import CheckpointEngine, SQLiteStore

adapter = LangGraphAdapter(CheckpointEngine(SQLiteStore("agent.db")))

async with adapter.run("execution-1") as run:
    await run.node("plan", node_type="PROMPT")
    # ... your langgraph logic
```

On crash, `RecoveryEngine` gives you a `RecoveryPlan` with exactly which nodes to replay and which to skip.

Zero deps (sqlite3, asyncio, stdlib only). Optional Redis/Postgres backends.

GitHub: https://github.com/likkisamarthreddy/livingai
PyPI: `pip install livingai`

---

## Post both at:
- https://reddit.com/r/Python/submit
- https://reddit.com/r/LangChain/submit
- https://reddit.com/r/LocalLLaMA/submit  (title: "Crash recovery library for local LLM agents")

**Best time:** Tuesday–Thursday, 8am–12pm UTC
