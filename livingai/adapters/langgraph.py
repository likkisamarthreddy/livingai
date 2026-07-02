"""LangGraph adapter.

Translates LangGraph execution events into :class:`ExecutionNode` records and
persists them through a :class:`CheckpointEngine`. It is a *thin* translation
layer:

* It does **not** import ``langgraph``. Events are consumed as plain data
  (dict-like), so the adapter runs anywhere and the core stays zero-dependency.
* A LangGraph node maps to a runtime node type: nodes whose name suggests a tool
  (or are explicitly flagged) become ``TOOL`` nodes and are marked
  **non-idempotent** so the recovery engine will not re-run their side effects.

Typical usage inside a LangGraph run::

    adapter = LangGraphAdapter(engine, execution_id="run-1")
    node = await adapter.on_node_start(name="retrieve", input={"q": "..."})
    ...
    await adapter.on_node_end(node, output=result, state=serialized_state)
"""

from __future__ import annotations

from typing import Optional

from ..graph import NodeType
from ._base import BaseAdapter, classify_name


__all__ = ["LangGraphAdapter", "classify"]


# Heuristic: LangGraph node names that look like tools default to non-idempotent.
_TOOL_HINTS = ("tool", "action", "call", "api", "http", "search", "fetch", "write")


def classify(name: str, *, node_type: Optional[NodeType] = None) -> NodeType:
    """Map a LangGraph node name to a runtime :class:`NodeType`.

    An explicit ``node_type`` always wins; otherwise the name is matched against
    tool-like hints, defaulting to ``PROMPT``.
    """
    return classify_name(name, _TOOL_HINTS, node_type=node_type)


class LangGraphAdapter(BaseAdapter):
    """Records a LangGraph execution as runtime execution nodes."""

    framework = "langgraph"
    node_key = "lg_node"
    tool_hints = _TOOL_HINTS

