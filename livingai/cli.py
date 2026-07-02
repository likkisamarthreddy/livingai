"""``livingai`` command-line interface.

Commands operate on a durable SQLite execution log:

* ``livingai list --db PATH`` — list execution ids.
* ``livingai show EXECUTION_ID --db PATH`` — print an execution's node graph.
* ``livingai replay EXECUTION_ID --db PATH [--mode ...] [--from NODE_ID]`` —
  replay a recorded execution. Without user-supplied executors the CLI performs
  a *reconstruction* replay: each node "re-executes" to its recorded output,
  which is exactly the MOCK_TOOLS semantics for tool nodes and a faithful
  dry-run for the rest. This is the debugging entry point from the plan:
  ``livingai replay <execution_id>``.

The functions are written to return strings so they are directly unit-testable;
``main`` handles argument parsing and printing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Optional, Sequence

from .graph import ExecutionNode
from .replay import ReplayMode, ReplaySession
from .storage import SQLiteStore


__all__ = ["main"]


async def _list_executions(store: SQLiteStore) -> list[str]:
    # Query distinct execution ids directly from the durable table.
    def _query() -> list[str]:
        with store._lock:  # noqa: SLF001 - CLI is part of the package
            rows = store._conn.execute(
                "SELECT execution_id FROM nodes"
                " GROUP BY execution_id ORDER BY MIN(seq)"
            ).fetchall()
        return [r["execution_id"] for r in rows]

    return await asyncio.to_thread(_query)


def cmd_list(db: str) -> str:
    store = SQLiteStore(db)
    try:
        ids = asyncio.run(_list_executions(store))
    finally:
        store.close()
    if not ids:
        return "(no executions found)"
    return "\n".join(ids)


def cmd_show(execution_id: str, db: str) -> str:
    store = SQLiteStore(db)
    try:
        nodes = asyncio.run(store.list_by_execution(execution_id))
    finally:
        store.close()
    if not nodes:
        return f"(no nodes for execution {execution_id})"
    lines = [f"Execution {execution_id} — {len(nodes)} node(s):"]
    for n in nodes:
        ckpt = "*" if n.checkpoint is not None else " "
        idem = "idem" if n.is_idempotent() else "side-effect"
        lines.append(
            f"  [{ckpt}] {n.type.value:<7} {n.status.value:<7} {idem:<11} {n.id}"
        )
    return "\n".join(lines)


def cmd_replay(
    execution_id: str,
    db: str,
    *,
    mode: ReplayMode = ReplayMode.MOCK_TOOLS,
    from_node_id: Optional[str] = None,
) -> str:
    store = SQLiteStore(db)

    async def reconstruct(node: ExecutionNode) -> Any:
        # Reconstruction replay: reproduce the recorded output.
        return node.output

    try:
        session = ReplaySession(store, execution_id)
        results = asyncio.run(
            session.run(reconstruct, mode=mode, from_node_id=from_node_id)
        )
    finally:
        store.close()

    if not results:
        return f"(nothing to replay for execution {execution_id})"
    lines = [f"Replayed {execution_id} in {mode.value} mode — {len(results)} node(s):"]
    for r in results:
        tag = "mock" if r.mocked else "run "
        lines.append(f"  {tag} {r.node.type.value:<7} {r.node.id} -> {r.output!r}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="livingai", description="Living AI runtime CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list execution ids in a store")
    p_list.add_argument("--db", required=True, help="path to the SQLite store")

    p_show = sub.add_parser("show", help="print an execution's node graph")
    p_show.add_argument("execution_id")
    p_show.add_argument("--db", required=True)

    p_replay = sub.add_parser("replay", help="replay a recorded execution")
    p_replay.add_argument("execution_id")
    p_replay.add_argument("--db", required=True)
    p_replay.add_argument(
        "--mode",
        choices=[m.value for m in ReplayMode],
        default=ReplayMode.MOCK_TOOLS.value,
    )
    p_replay.add_argument("--from", dest="from_node_id", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        print(cmd_list(args.db))
    elif args.command == "show":
        print(cmd_show(args.execution_id, args.db))
    elif args.command == "replay":
        print(
            cmd_replay(
                args.execution_id,
                args.db,
                mode=ReplayMode(args.mode),
                from_node_id=args.from_node_id,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
