# CLI

The `livingai` command operates on a durable SQLite execution log. Install the
package (`pip install -e .`) to register it, or run it as a module:

```bash
livingai --help
python -m livingai.cli --help
```

Every command requires `--db PATH` pointing at a `SQLiteStore` file (the log your
application wrote to).

## `list`

List the execution ids in a store.

```bash
livingai list --db agent.db
```

## `show`

Print an execution's node graph. A `*` marks nodes carrying a checkpoint, and
each node shows whether it is idempotent or a side-effecting call.

```bash
livingai show run-1 --db agent.db
```

```
Execution run-1 — 3 node(s):
  [*] PROMPT  SUCCESS idem        a49ca6c2-...
  [ ] TOOL    SUCCESS side-effect ae0f8fba-...
  [ ] PROMPT  SUCCESS idem        1b2c3d4e-...
```

## `replay`

Replay a recorded execution. Without user-supplied executors the CLI performs a
**reconstruction replay**: each node reproduces its recorded output — exactly the
`MOCK_TOOLS` semantics for tool nodes and a faithful dry-run for the rest.

```bash
livingai replay run-1 --db agent.db --mode MOCK_TOOLS
livingai replay run-1 --db agent.db --mode FROM_NODE --from <node_id>
```

```
Replayed run-1 in MOCK_TOOLS mode — 3 node(s):
  run  PROMPT  a49ca6c2-... -> 'thought'
  mock TOOL    ae0f8fba-... -> {'temp': 21}
  run  PROMPT  1b2c3d4e-... -> 'Sunny 21C'
```

`--mode` accepts `FULL`, `FROM_NODE`, `MOCK_TOOLS`, or `COUNTERFACTUAL`
(default `MOCK_TOOLS`).
