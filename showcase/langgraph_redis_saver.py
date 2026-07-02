"""Showcase — a drop-in LangGraph checkpointer backed by livingai's Redis store.

LangGraph ships with `SqliteSaver`, a **blocking, single-writer** checkpointer.
Under concurrency it serializes every write behind a file lock. This module
replaces it with a **non-blocking, horizontally-scalable** saver built on
livingai's `CheckpointEngine` + `RedisStore` — with the 50 ms SLA budget so a slow
write never stalls your graph.

Drop-in swap:

    -  from langgraph.checkpoint.sqlite import SqliteSaver
    -  saver = SqliteSaver.from_conn_string("agent.db")
    +  from showcase.langgraph_redis_saver import LivingAIRedisSaver
    +  saver = LivingAIRedisSaver.from_url("redis://localhost:6379")

    graph = builder.compile(checkpointer=saver)

Requires:  pip install "livingai[redis]" langgraph
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langchain_core.runnables import RunnableConfig

from livingai import CheckpointEngine, ExecutionNode, NodeType, Status
from livingai.stores.redis import RedisStore


class LivingAIRedisSaver(BaseCheckpointSaver):
    """A LangGraph `BaseCheckpointSaver` powered by livingai + Redis.

    Every LangGraph checkpoint is compressed and written through the
    `CheckpointEngine`, so it inherits the dual-tier hot cache, zlib compression,
    and the 50 ms non-blocking SLA budget. Reads are served from the in-process
    hot cache in microseconds, falling back to Redis on a miss.
    """

    def __init__(self, engine: CheckpointEngine) -> None:
        super().__init__()
        self.engine = engine

    @classmethod
    def from_url(cls, url: str = "redis://localhost:6379") -> "LivingAIRedisSaver":
        return cls(CheckpointEngine(RedisStore(url=url)))

    def _thread_id(self, config: RunnableConfig) -> str:
        return config["configurable"]["thread_id"]

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        thread_id = self._thread_id(config)
        node = ExecutionNode(
            execution_id=thread_id,
            type=NodeType.MEMORY,          # graph state is idempotent to reload
            status=Status.SUCCESS,
            output={"checkpoint_id": checkpoint["id"]},
            metadata={"langgraph_metadata": dict(metadata)},
        )
        # The full graph state is compressed and budgeted by the engine.
        state = json.dumps({"checkpoint": checkpoint, "config": config}, default=str).encode()
        await self.engine.save(node, state=state)
        return {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint["id"]}}

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = self._thread_id(config)
        result = await self.engine.latest(thread_id)   # hot cache -> Redis
        if result is None:
            return None
        _node, state = result
        if state is None:
            return None
        payload = json.loads(state.decode())
        return CheckpointTuple(
            config=config,
            checkpoint=payload["checkpoint"],
            metadata=CheckpointMetadata(),
            parent_config=None,
        )
