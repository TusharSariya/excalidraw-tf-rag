# pipeline-v2

Phase P1 of the distributed rearchitecture (see
`../docs/rearchitecture-v2-decision-log.md` and
`../docs/distributed-rearchitecture-rfc.md`). Two pieces land in this phase:

1. **Work-state ledger** (`pipeline_v2.workstate`) — the Postgres-backed
   per-(doc, corpus, stage) outcome ledger that replaces the v1 bug class
   where a `degraded` outcome could still be read back as "done" and get
   indexed. `state` is a hard `CHECK (state IN ('ok', 'degraded', 'failed'))`
   and the `indexable_work_state` view (plus every read helper in this
   package) only ever surfaces `state = 'ok'` rows.
2. **Manifest migration** (`pipeline_v2.manifest_migrate`) — imports a v1
   `manifest.json` (dict-level, no v1 code imported) into a `manifest_item`
   table, and exports it back out with a byte-level round-trip check against
   a canonicalized (sorted-keys) form of the input.

## Commands

```bash
uv sync
uv run pipeline-v2 workstate init-db
uv run pipeline-v2 workstate retry-dead-letter --stage embed --corpus graph-layout-rag
uv run pipeline-v2 manifest migrate --json ../graph-layout-rag/data/harvest/manifest.json --corpus graph-layout-rag --verify
uv run pipeline-v2 manifest export --corpus graph-layout-rag --out /tmp/manifest.export.json
```

`PIPELINE_V2_DATABASE_URL` selects the Postgres DSN (defaults to
`postgresql://postgres:postgres@localhost:5432/pipeline_v2`).

## Tests

```bash
uv sync
uv run pytest                       # offline unit subset (in-memory repo only)
RUN_INTEGRATION=1 uv run pytest     # + real postgres:16 via docker (session-scoped container)
```
