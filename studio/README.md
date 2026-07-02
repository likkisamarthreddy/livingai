# Living AI Studio — the visual dashboard

A command-line tool tells you *what* happened. **Studio shows you** — and lets you
rewind.

Studio reads any livingai SQLite store and renders every execution as an
interactive graph: which nodes succeeded (green), which failed (red), which have
side effects (boxes) — plus a **"Replay from this node"** button that rewinds the
agent and re-runs it safely in `FROM_NODE` mode.

> Watching an agent *rewind* and re-execute a step differently is the "aha!"
> moment that makes developers adopt the runtime.

## Quickstart

```bash
pip install "livingai[studio]"          # streamlit
python studio/seed_demo.py              # creates studio_demo.db with sample runs
streamlit run studio/app.py -- --db studio_demo.db
```

Then open the browser tab Streamlit prints (usually http://localhost:8501).

## What you get

- **Execution picker** — every recorded run in the store, in the sidebar.
- **Live graph** — node types (`PROMPT` ellipse, `TOOL` box), colored by status:
  🟢 SUCCESS · 🔴 FAILED · 🟡 PENDING. The currently-selected node is outlined.
- **Summary metrics** — node count, successes, failures, and how many nodes have
  side effects (non-idempotent).
- **Node inspector** — expand any node to see its output, token cost, and id.
- **⏪ Replay from this node** — re-runs the execution from that point, returning
  recorded tool outputs so no real API calls or charges fire.

## Point it at your own database

```bash
streamlit run studio/app.py -- --db /path/to/your/agent.db
```

Or paste the path into the sidebar while it's running.

## Files

- [`app.py`](app.py) — the Streamlit dashboard.
- [`seed_demo.py`](seed_demo.py) — populates `studio_demo.db` with example runs.
