"""CrewAI adapter.

Records a CrewAI run as runtime execution nodes. Like the LangGraph adapter it
imports **no** ``crewai`` package — task/agent events are consumed as plain data.

CrewAI vocabulary maps naturally onto the runtime model: an agent's *task* is a
node, and steps that invoke *tools* or *delegate* to another agent are treated as
side-effecting (non-idempotent) ``TOOL`` nodes.

    adapter = CrewAIAdapter(engine, execution_id="crew-run")
    node = await adapter.on_node_start(name="research_task")
    await adapter.on_node_end(node, output=result, state=serialized_state)
"""

from __future__ import annotations

from ._base import BaseAdapter


__all__ = ["CrewAIAdapter"]


class CrewAIAdapter(BaseAdapter):
    """Records a CrewAI execution as runtime execution nodes."""

    framework = "crewai"
    node_key = "crew_node"
    tool_hints = (
        "tool",
        "delegate",
        "handoff",
        "execute",
        "call",
        "api",
        "search",
        "scrape",
    )
