# Discord / Community Posts

## LangGraph Discord  (#tools or #showcase channel)

> **Hey all** — just shipped an open-source crash recovery library for AI agents.
>
> The core problem it solves: when your agent crashes mid-run, you want to resume from the last checkpoint without re-running non-idempotent tool calls (payments, emails, writes).
>
> **livingai** records every step to an append-only log and builds a recovery plan that explicitly separates "safe to replay" (LLM calls, memory reads) from "must skip" (tool side effects).
>
> Also has a `MOCK_TOOLS` replay mode — re-run a recorded execution with stored tool responses for zero-cost debugging.
>
> Zero deps, works alongside any LangGraph setup, Apache-2.0.
> `pip install livingai`
> GitHub: https://github.com/likkisamarthreddy/livingai
>
> Would love feedback from anyone using persistent agents in production.

---

## CrewAI Discord  (#tools or #integrations channel)

> **Shipped:** crash recovery + replay for CrewAI workflows — `pip install livingai`
>
> When a CrewAI crew crashes mid-execution, `livingai` lets you resume from the last durable checkpoint. The recovery engine knows which crew tool calls had side effects and skips them during replay.
>
> Has a thin `CrewAIAdapter` — wraps around your existing crew execution with 3 extra lines.
> GitHub: https://github.com/likkisamarthreddy/livingai

---

## X (Twitter) — 3-post thread

**Post 1:**
AI agents fail in expensive ways. After the LLM reasoned. After the tool charged the card. Three steps into a ten-step plan.

I built `livingai` — crash recovery and replay for agents.

Zero deps. `pip install livingai`

🧵

**Post 2:**
The recovery engine knows which nodes are safe to replay.

PROMPT + MEMORY nodes (idempotent) → replay
TOOL nodes with side effects (card, email, write) → skip forever

Your card is never charged twice during crash recovery.

github.com/likkisamarthreddy/livingai

**Post 3:**
The debugging superpower: `MOCK_TOOLS` replay.

Record a run once. Replay it 100 times returning stored responses.
Zero API calls. Zero cost. Full reasoning visible.

Works with LangGraph, CrewAI, OpenAI Agents SDK.

`pip install livingai`

---

## LinkedIn post

**I just open-sourced livingai — crash recovery and checkpointing for AI agents.**

The problem: AI agents are expensive workflows. An LLM call costs money. A tool call can charge a card, send an email, create a record. When the process crashes mid-execution, naïvely restarting re-runs all of that.

livingai records every agent step to an append-only log and builds a recovery plan that is explicit about idempotency: LLM calls are safe to replay, tool side effects are not.

Key features:
→ Resume from last checkpoint after crash
→ MOCK_TOOLS mode: re-run with stored responses (zero API cost)
→ COUNTERFACTUAL mode: re-run with modified inputs
→ LangGraph, CrewAI, OpenAI Agents adapters
→ SQLite, Redis, and PostgreSQL backends
→ Zero runtime dependencies
→ 128 tests, 100% coverage, mypy --strict, CI 3.9–3.12

`pip install livingai`

GitHub: https://github.com/likkisamarthreddy/livingai

If you're building agents in production, I'd love to hear how recovery fits (or doesn't fit) your workflow.

#python #aiagents #llm #opensource #langchain #langgraph
