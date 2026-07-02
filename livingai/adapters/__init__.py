"""Framework adapters — thin translation layers.

An adapter maps a framework's events onto the runtime's :class:`ExecutionNode`
model. Per the plan's *framework-agnostic core* principle, adapters live outside
the core and never leak framework types into it. All adapters share
:class:`BaseAdapter`.
"""

from ._base import BaseAdapter
from .langgraph import LangGraphAdapter
from .crewai import CrewAIAdapter
from .openai_agents import OpenAIAgentsAdapter

__all__ = [
    "BaseAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
]
