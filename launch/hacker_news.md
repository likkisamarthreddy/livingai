# Hacker News — "Show HN" post

**Title:**
Show HN: Living AI – crash recovery and replay for AI agents (zero deps, LangGraph/CrewAI/OpenAI)

**Body:**
AI agents fail in expensive ways. A process crashes after the LLM reasoned, after the tool charged a card, three steps into a ten-step plan — all the work is gone. On restart you re-pay for the tokens and hope the tool doesn't fire its side effects twice.

I built Living AI to fix this. It records every step of an agent execution to an append-only log so any run can be recovered or replayed without re-running side effects.

The recovery logic is explicit about idempotency: TOOL nodes default to non-idempotent, so a card charge or email send is never re-executed during crash recovery. Only safe work is replayed.

---

**What it does:**
- Checkpoint after every agent step (50ms overhead budget enforced in code — if the write takes longer it's dropped rather than blocking the agent)
- On crash: resume from the last durable checkpoint, replay only idempotent work
- MOCK_TOOLS replay mode: re-run a recorded execution returning stored tool responses — iterate on reasoning without real API calls
- COUNTERFACTUAL mode: re-run with modified inputs to understand what would have changed
- Works with LangGraph, CrewAI, and the OpenAI Agents SDK through thin adapters

**Install:**
```
pip install livingai
```

Zero runtime dependencies (pure stdlib: sqlite3, asyncio, zlib, dataclasses, uuid).

Optional Redis and PostgreSQL backends: `pip install livingai[redis]` / `livingai[postgres]`

**GitHub:** https://github.com/likkisamarthreddy/livingai
**PyPI:** https://pypi.org/project/livingai/

---

**Post at:** https://news.ycombinator.com/submit
**Best time:** Tuesday–Thursday, 9–11am US Eastern
