"""Living AI runtime — checkpoint, recovery, and replay for AI agents.

Zero-dependency core package.

Public surface by phase:

* **Phase 0** — execution graph data model and storage protocol with a default
  zero-config SQLite backend.
* **Phase 1** — checkpoint engine (compression, hot-tier caching, and hard
  overhead-budget enforcement) plus metrics.
* **Phase 2** — recovery engine (crash recovery with idempotent replay) and the
  first framework adapter (LangGraph).
* **Phase 3** — replay engine (FULL / FROM_NODE / MOCK_TOOLS / COUNTERFACTUAL)
  and the ``livingai`` CLI.
* **Phase 5** — optional Redis and PostgreSQL backends
  (``pip install livingai[redis]`` / ``livingai[postgres]``).
"""

from __future__ import annotations

from .graph import (
    ErrorInfo,
    ExecutionNode,
    NodeType,
    Status,
    new_id,
    utcnow,
)
from .storage import CheckpointStore, SQLiteStore
from .compression import Compressor, NoopCompressor, ZlibCompressor
from .metrics import Metrics
from .checkpoint import CheckpointEngine, HotCache
from .recovery import RecoveryEngine, RecoveryPlan
from .replay import ReplayMode, ReplayResult, ReplaySession
from .adapters import (
    BaseAdapter,
    LangGraphAdapter,
    CrewAIAdapter,
    OpenAIAgentsAdapter,
)

__version__ = "0.4.0"

__all__ = [
    # graph
    "ErrorInfo",
    "ExecutionNode",
    "NodeType",
    "Status",
    "new_id",
    "utcnow",
    # storage
    "CheckpointStore",
    "SQLiteStore",
    # compression
    "Compressor",
    "NoopCompressor",
    "ZlibCompressor",
    # checkpoint engine
    "CheckpointEngine",
    "HotCache",
    "Metrics",
    # recovery
    "RecoveryEngine",
    "RecoveryPlan",
    # replay
    "ReplayMode",
    "ReplayResult",
    "ReplaySession",
    # adapters
    "BaseAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "__version__",
]
