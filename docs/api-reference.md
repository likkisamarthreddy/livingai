# API Reference

The complete public surface, importable from the top-level `livingai` package.

```python
from livingai import (
    # graph
    ExecutionNode, NodeType, Status, ErrorInfo, new_id, utcnow,
    # storage
    CheckpointStore, SQLiteStore,
    # compression
    Compressor, ZlibCompressor, NoopCompressor,
    # checkpoint engine
    CheckpointEngine, HotCache, Metrics,
    # recovery
    RecoveryEngine, RecoveryPlan,
    # replay
    ReplaySession, ReplayMode, ReplayResult,
    # adapters
    LangGraphAdapter,
)
```

## Graph

### `ExecutionNode`
Dataclass. Fields: `execution_id`, `type`, `id`, `parent_id`, `status`,
`created_at`, `completed_at`, `input`, `output`, `error`, `cost_tokens`,
`latency_ms`, `metadata`, `checkpoint`.

- `is_idempotent() -> bool` — safe to re-run during recovery?
- `to_dict()` / `from_dict(data, checkpoint=None)` — JSON-safe mapping (excludes
  the binary checkpoint).
- `to_json()` / `from_json(raw, checkpoint=None)` — deterministic JSON string.

### `NodeType`
Enum: `PROMPT`, `TOOL`, `MEMORY`, `BRANCH`.

### `Status`
Enum: `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`.

### `ErrorInfo`
Dataclass: `type`, `message`, `traceback?`. `to_dict()` / `from_dict()`.

### helpers
- `new_id() -> str` — a UUID4 string.
- `utcnow() -> datetime` — timezone-aware UTC now.

## Storage

### `CheckpointStore` (Protocol)
```python
async def write(node: ExecutionNode) -> None
async def read(node_id: str) -> ExecutionNode | None
async def list_by_execution(execution_id: str) -> list[ExecutionNode]
async def get_latest_checkpoint(execution_id: str) -> ExecutionNode | None
```
Implementations are append-only.

### `SQLiteStore(path=":memory:")`
Zero-config default backend. `close()` releases the connection.

## Compression

### `Compressor` (Protocol)
`compress(data: bytes) -> bytes`, `decompress(blob: bytes) -> bytes`.

### `ZlibCompressor(level=6)`
Standard-library default. Self-describing codec header.

### `NoopCompressor`
Pass-through codec.

## Checkpoint engine

### `CheckpointEngine(store, *, overhead_budget_ms=50, compressor=None, hot_capacity=128, hot_ttl_seconds=3600, metrics=None)`
- `async save(node, state: bytes | None = None) -> bool`
- `async load(node_id) -> tuple[ExecutionNode, bytes | None] | None`
- `async latest(execution_id) -> tuple[ExecutionNode, bytes | None] | None`
- `.metrics: Metrics`, `.store`, `.hot: HotCache`

### `HotCache(capacity=128, ttl_seconds=3600)`
- `put(node)`, `get(node_id) -> ExecutionNode | None`, `len()`

### `Metrics()`
- `increment(name, amount=1)`, `observe(name, value)`
- `counter(name) -> int`, `samples(name) -> list[float]`
- `percentile(name, pct) -> float | None`, `snapshot() -> dict`

## Recovery

### `RecoveryEngine(checkpoint_engine, *, metrics=None)`
- `async plan(execution_id) -> RecoveryPlan`
- `async replay(plan, handler) -> list`
- `async recover(execution_id, handler) -> RecoveryPlan`

### `RecoveryPlan` (frozen dataclass)
`execution_id`, `found`, `checkpoint_node`, `state`, `replay_nodes`,
`skipped_nodes`, `resume_node_id`.

## Replay

### `ReplaySession(store, execution_id, *, metrics=None)`
```python
async def run(handler, *, mode=ReplayMode.FULL,
              from_node_id=None, counterfactual=None) -> list[ReplayResult]
```

### `ReplayMode`
Enum: `FULL`, `FROM_NODE`, `MOCK_TOOLS`, `COUNTERFACTUAL`.

### `ReplayResult`
Dataclass: `node`, `output`, `mocked`.

## Adapters

### `LangGraphAdapter(engine, execution_id=None)`
- `async on_node_start(name, *, input=None, node_type=None, parent_id=None, idempotent=None, metadata=None) -> ExecutionNode`
- `async on_node_end(node, *, output=None, state=None, cost_tokens=None) -> bool`
- `async on_node_error(node, error) -> bool`

## CLI

`livingai.cli.main(argv=None) -> int`. Subcommands: `list`, `show`, `replay`.
See [CLI](cli.md).
