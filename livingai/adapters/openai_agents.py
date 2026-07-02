"""OpenAI Agents SDK adapter.

Records an OpenAI Agents SDK run as runtime execution nodes. Imports **no**
``openai`` package — agent/tool events are consumed as plain data.

The Agents SDK expresses work as the agent's reasoning turns plus *function*
(tool) calls and *handoffs* to other agents. Function/tool calls are treated as
side-effecting (non-idempotent) ``TOOL`` nodes so recovery won't re-invoke them.

    adapter = OpenAIAgentsAdapter(engine, execution_id="oa-run")
    node = await adapter.on_node_start(name="get_weather_function")
    await adapter.on_node_end(node, output=result, cost_tokens=120)
"""

from __future__ import annotations

from ._base import BaseAdapter


__all__ = ["OpenAIAgentsAdapter"]


class OpenAIAgentsAdapter(BaseAdapter):
    """Records an OpenAI Agents SDK execution as runtime execution nodes."""

    framework = "openai-agents"
    node_key = "agent_node"
    tool_hints = (
        "tool",
        "function",
        "call",
        "api",
        "handoff",
        "retrieval",
        "search",
        "fetch",
    )
