"""Living AI Studio — a visual dashboard for agent executions.

Reads a livingai SQLite store and renders every recorded execution as an
interactive graph: which nodes succeeded, which failed, which had side effects —
and a **"Replay from this node"** button that rewinds the agent and re-runs it in
`MOCK_TOOLS` mode (no real tool calls, no double billing).

Run:

    pip install "livingai[studio]"        # installs streamlit + graphviz
    streamlit run studio/app.py -- --db agent.db

Or point it at any database from the sidebar.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import streamlit as st

from livingai import ExecutionNode, NodeType, RecoveryEngine, CheckpointEngine, SQLiteStore
from livingai.replay import ReplayMode, ReplaySession


# --------------------------------------------------------------------------- #
# Data access (async store, run synchronously for Streamlit).
# --------------------------------------------------------------------------- #
def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def list_executions(store: SQLiteStore) -> list[str]:
    def _q() -> list[str]:
        with store._lock:  # noqa: SLF001
            rows = store._conn.execute(
                "SELECT execution_id FROM nodes GROUP BY execution_id ORDER BY MIN(seq)"
            ).fetchall()
        return [r["execution_id"] for r in rows]

    return _run(asyncio.to_thread(_q))


def load_nodes(store: SQLiteStore, execution_id: str) -> list[ExecutionNode]:
    return _run(store.list_by_execution(execution_id))


def replay_from(store: SQLiteStore, execution_id: str, node_id: str) -> list:
    async def reconstruct(node: ExecutionNode):
        return node.output

    session = ReplaySession(store, execution_id)
    return _run(
        session.run(reconstruct, mode=ReplayMode.FROM_NODE, from_node_id=node_id)
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_STATUS_COLOR = {"SUCCESS": "#16a34a", "FAILED": "#dc2626", "PENDING": "#a16207"}


def build_graph(nodes: list[ExecutionNode], highlight: str | None) -> str:
    """Return Graphviz DOT for the execution."""
    lines = ["digraph G {", '  rankdir=TB;', '  node [style=filled, fontname="Helvetica"];']
    prev = None
    for n in nodes:
        color = _STATUS_COLOR.get(n.status.value, "#6b7280")
        shape = "box" if n.type is NodeType.TOOL else "ellipse"
        border = "3" if n.id == highlight else "1"
        label = f"{n.type.value}\\n{n.output if n.output else ''}".replace('"', "'")[:40]
        lines.append(
            f'  "{n.id}" [label="{label}", fillcolor="{color}", fontcolor=white, '
            f'shape={shape}, penwidth={border}];'
        )
        if prev is not None:
            lines.append(f'  "{prev}" -> "{n.id}";')
        prev = n.id
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="agent.db")
    args, _ = parser.parse_known_args(sys.argv[1:])

    st.set_page_config(page_title="Living AI Studio", page_icon="🧠", layout="wide")
    st.title("🧠 Living AI Studio")
    st.caption("Visualize, inspect, and replay agent executions.")

    db_path = st.sidebar.text_input("SQLite store path", value=args.db)

    try:
        store = SQLiteStore(db_path)
    except Exception as e:  # pragma: no cover - UI guard
        st.error(f"Could not open store: {e}")
        return

    executions = list_executions(store)
    if not executions:
        st.info(f"No executions found in `{db_path}`. Run an agent first.")
        return

    execution_id = st.sidebar.selectbox("Execution", executions)
    nodes = load_nodes(store, execution_id)

    # Summary metrics.
    ok = sum(1 for n in nodes if n.status.value == "SUCCESS")
    failed = sum(1 for n in nodes if n.status.value == "FAILED")
    side_effects = sum(1 for n in nodes if not n.is_idempotent())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", len(nodes))
    c2.metric("Succeeded", ok)
    c3.metric("Failed", failed)
    c4.metric("Side-effect nodes", side_effects)

    highlight = st.session_state.get("highlight")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Execution graph")
        st.graphviz_chart(build_graph(nodes, highlight))

    with right:
        st.subheader("Nodes")
        for n in nodes:
            idem = "idempotent" if n.is_idempotent() else "⚠️ side-effect"
            with st.expander(f"{n.type.value} · {n.status.value} · {idem}"):
                st.code(f"id: {n.id}\noutput: {n.output}\ncost_tokens: {n.cost_tokens}")
                if st.button("⏪ Replay from this node", key=f"replay-{n.id}"):
                    st.session_state["highlight"] = n.id
                    results = replay_from(store, execution_id, n.id)
                    st.success(f"Replayed {len(results)} node(s) in MOCK_TOOLS-safe mode.")
                    for r in results:
                        tag = "mock" if getattr(r, "mocked", False) else "run"
                        st.write(f"`{tag}` **{r.node.type.value}** → {r.output!r}")


if __name__ == "__main__":
    main()
