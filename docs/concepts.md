# Concepts

## The execution graph

Every agent execution produces a directed acyclic graph (DAG) of
`ExecutionNode` records. A node is the atomic unit of work — a prompt, a tool
call, a memory read, or a branch.

```
ExecutionNode
├── id            str        globally unique
├── parent_id     str?       null for the root
├── execution_id  str        groups all nodes in one run
├── type          NodeType   PROMPT | TOOL | MEMORY | BRANCH
├── status        Status     PENDING | RUNNING | SUCCESS | FAILED
├── created_at    datetime   UTC, timezone-aware
├── completed_at  datetime?
├── input         JSON
├── output        JSON?
├── error         ErrorInfo? {type, message, traceback}
├── cost_tokens   int?
├── latency_ms    int?
├── metadata      JSON       framework-specific data
└── checkpoint    bytes?     compressed serialized state
```

`input`, `output`, and `metadata` are stored as JSON blobs rather than typed
columns. This keeps the schema stable across framework and product versions;
schema evolution is an application concern, not a database migration.

## Append-only log

The log is **never mutated** — only appended to. Advancing a node's status
(`PENDING → RUNNING → SUCCESS`) is done by writing the node again; the store
keeps every version and returns the latest projection on read.

Why it matters: an append-only log means the state of any execution at any point
in time can be reconstructed deterministically. It is immune to corruption on
crashes and is the foundation for recovery, replay, and auditing.

## Idempotency

Recovery must never re-trigger external side effects (payments, emails, API
writes). Each node reports whether re-execution is safe via
`ExecutionNode.is_idempotent()`:

1. An explicit `metadata["idempotent"]` boolean always wins.
2. Otherwise `TOOL` nodes default to **non-idempotent**; all other types default
   to idempotent.

Adapters annotate specific tool calls (e.g. a read-only API can be marked
`idempotent=True`).

## Storage tiers

The runtime uses two tiers with different performance profiles:

- **Tier 1 — hot cache** (`HotCache`): an in-process LRU + TTL cache of the most
  recent checkpoints, sized for sub-millisecond recovery reads.
- **Tier 2 — cold store** (`CheckpointStore`): the durable, append-only history.
  `SQLiteStore` is the zero-config default; Redis/Postgres backends are planned.

The `CheckpointEngine` writes through both and reads hot-first, cold-fallback.

## Components at a glance

```
ExecutionNode ──► CheckpointStore (Tier 2, durable)
      ▲                  ▲
      │                  │
 Adapters          CheckpointEngine ──► HotCache (Tier 1)
 (LangGraph)             │
                    RecoveryEngine ──► RecoveryPlan
                    ReplaySession  ──► ReplayResult
```
