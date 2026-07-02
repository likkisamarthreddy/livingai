# Contributing to Living AI Runtime

Thanks for your interest in contributing! This is an infrastructure project, so
the bar is **correctness and reliability first**. Boring, well-tested code wins.

## Development setup

```bash
cd livingai_runtime
python -m venv .venv && . .venv/bin/activate    # or your preferred env
pip install -e ".[dev]"
pip install mypy
```

The core has **zero runtime dependencies** — only the standard library. Do not
add a runtime dependency to `livingai/` without discussion; it undermines the
zero-config promise.

## Running the checks locally

Everything CI runs, you can run locally:

```bash
# Tests
python -m pytest -q

# Coverage (must stay at 100%)
python -m coverage run -m pytest -q
python -m coverage report -m --include="*/livingai/*" --fail-under=100

# Strict type checking (must pass clean)
python -m mypy --strict livingai

# Benchmarks (sanity check performance)
python benchmarks/benchmark.py
```

## Standards

- **100% test coverage** on `livingai/`. New code needs new tests.
- **`mypy --strict` clean.** Full type hints; the package ships `py.typed`.
- **Async-first.** All I/O is `async`; provide sync wrappers only when needed.
- **Append-only.** Never mutate stored records; write a new node version.
- **No global state.** Engines and stores are independent instances.
- Tests use plain `asyncio.run(...)` (no `pytest-asyncio` dependency).

## Adding a new framework adapter

Adapters are thin translation layers. Subclass
[`BaseAdapter`](livingai/adapters/_base.py) and set three attributes:

```python
from livingai.adapters._base import BaseAdapter

class MyFrameworkAdapter(BaseAdapter):
    framework = "myframework"          # metadata["framework"] tag
    node_key = "mf_node"               # metadata key for the node name
    tool_hints = ("tool", "call", ...) # names that mark side-effecting TOOL nodes
```

Rules:

- **Do not import the framework package.** Consume events as plain data so the
  adapter runs anywhere and the core stays dependency-free.
- Mark side-effecting steps as `TOOL` (auto non-idempotent) so recovery never
  re-runs them. Allow explicit `idempotent=` overrides.
- Add tests mirroring [`tests/test_more_adapters.py`](tests/test_more_adapters.py).
- Export it from `livingai/adapters/__init__.py` and the top-level `livingai`.

## Extending the storage protocol

New backends (Redis, Postgres, ...) implement the
[`CheckpointStore`](livingai/storage/__init__.py) protocol:

```python
async def write(node) -> None            # append-only
async def read(node_id) -> ExecutionNode | None
async def list_by_execution(execution_id) -> list[ExecutionNode]
async def get_latest_checkpoint(execution_id) -> ExecutionNode | None
```

A new backend must pass the **same** test suite as `SQLiteStore`
([`tests/test_sqlite_store.py`](tests/test_sqlite_store.py)) — parametrize the
store fixture rather than duplicating tests.

## Pull request checklist

- [ ] Tests added/updated; `pytest` green.
- [ ] Coverage at 100%.
- [ ] `mypy --strict` clean.
- [ ] Public API changes reflected in `docs/` and the top-level `__all__`.
- [ ] No new runtime dependency in `livingai/` (unless discussed).
- [ ] Commit messages are clear and scoped.

## Reporting bugs

Prefer a minimal reproduction using the in-memory `SQLiteStore()`. Include the
Python version and the exact steps. For anything involving data loss or
corruption, please flag it clearly — that is the highest-priority class of bug.
