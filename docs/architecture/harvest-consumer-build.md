# Harvest consumer build (Phase 1: compose-now, KEDA-later)

**Status:** approved design, ready to plan.
**Scope:** build the **consumer side** of the harvester — a seeder and a per-source
worker — packaged as one Docker container per queue, run by hand today, KEDA later.
Harvest only (discover → fetch source metadata → persist). **No text extraction.**

This spec is the concrete, buildable **delta** on the two canonical docs; it does
not restate them:

- `docs/architecture/harvest-worker-scaling.md` — **canonical** for worker/scaling
  semantics (drain loop, visibility timeout, poison cap, no-DLQ, resource sizing).
- `docs/architecture/harvest-implementation.md` — background. **Stale where it
  disagrees with the applied migrations** — notably its `source_arxiv` DDL predates
  `20260711011845_create_source_tables.sql` (real tables have a `status
  source_status` column and all-nullable data columns). Trust the migration.

---

## Decisions locked (this build)

| Decision | Choice | Rationale |
|---|---|---|
| Consumer shape | **One `worker.py`, parameterized by `--source`** | Drain/ack/retry/poison logic written once; per-source differs only in a fetch+upsert handler (`HARVESTERS` registry). "A consumer per source" is operational: N containers, each `harvest-work --source X`. |
| Lifecycle | **Drain-then-exit** (`exit 0` on empty queue) | Identical today (compose) and later (KEDA `ScaledJob`). No daemon/poll mode. |
| Packaging | **One container per queue**, `restart: "no"` | `docker compose` with a worker service per source. Brought up on demand; drains and exits. |
| Staging trigger | **Manual now → cron later → KEDA later** | Lowest commitment while building; same containers throughout. |
| DB backend | **Supabase** (staging, session pooler :5432) + local stack for dev | Portable by construction (direct psycopg + plain SQL + pgmq); a later move to self-hosted CloudNativePG is a `pg_dump` + DSN swap, near-zero code change. |
| DB access | **Direct `psycopg`** as an existing privileged role via a single `HARVEST_DB_DSN` | The fan-out path never uses PostgREST. One DSN env var — no pooler-URL reconstruction. |
| Rate limiting | **In-process `time.sleep` per source** (no shared bucket) | Single-threaded → concurrency 1 everywhere → the `rate_budget`/`take_token` bucket is not *needed* until a source runs >1 pod. **Deferred**, returns with KEDA scale-out. |
| DB roles | **Deferred** — use existing credentials now | The `harvester_worker` / `harvester_keda_ro` least-privilege split is prod/KEDA hardening; buys nothing under hand-run compose. Lands with KEDA. |
| openalex worker | **Not in this build** (no `OpenAlexSource` client yet) | 2 worker services now (arxiv, s2). `publish_seed` still fans out to `harvest_openalex`; those messages wait harmlessly in pgmq until the client + 3rd service land. |
| Message contract | **Versioned JSON Schema**, enforced in-DB + loaded by Python | One canonical `seed_message.v1.json`; the DB validates every message in `publish_seed` (via `pg_jsonschema`), Python consumers parse against the same file. `v` field lets the payload evolve (thin v1 → fat v2) without breaking consumers. |

**One new migration** (`20260711040000_enforce_seed_message_schema.sql`) for the
message contract; see below. The rest of the producer side (`seeds`,
`queue_subscriptions`, `publish_seed`, the three `harvest_*` queues, `source_*`
tables) already exists and is applied.

---

## Module layout (`v3/harvest/src/harvest/`)

```
__init__.py          # package init / logging only — legacy main() DELETED
settings.py          # + corpus, doi, source, harvest_db_dsn
db.py                # NEW — psycopg helpers
seeder.py            # NEW — harvest-seed  entrypoint
worker.py            # NEW — harvest-work --source X  entrypoint
contracts/           # NEW — the seed->message wire contract
  __init__.py        #   SCHEMA + parse() (tolerant reader); re-exports SeedMessage
  seed_message.v1.json  # canonical JSON Schema (DB enforces the same one)
  _seed_message_v1.py   # SeedMessage — GENERATED from the .json (do not hand-edit)
scripts/
  regenerate-contracts.sh  # regen _seed_message_v1.py from the .json (pinned)
sources/
  base.py
  arxiv.py           # fixed: strip stray debug + unreachable code; add typed accessor
  semantic_scholar.py# fixed: pprint NameError
  # openalex.py      later
```

### Message contract (`contracts/`) — **built**
The seed→message wire format is codified once, portably, and enforced on both ends:
- **Canonical:** `contracts/seed_message.v1.json` — a versioned JSON Schema.
  v1 is the thin envelope `{"v": 1, "paper_id": <bigint>}`. **Open content model**
  (no `additionalProperties: false`) on purpose — see the tolerant-reader rule below.
- **DB (producer) enforcement:** migration `20260711040000` adds `pg_jsonschema`,
  stores the schema as `public.seed_message_schema()`, and makes `publish_seed`
  build `{"v":1,"paper_id":…}` and `jsonb_matches_schema(...)`-validate it before
  every `pgmq.send` — a payload violating the contract (e.g. a null `paper_id`)
  raises instead of shipping.
- **Python (consumer) exposure:** `SeedMessage` is **generated** from the `.json`
  (`_seed_message_v1.py`, via `scripts/regenerate-contracts.sh`, generator pinned);
  `contracts/__init__.py` re-exports it plus `SCHEMA` and `parse()`. The worker does
  `contracts.parse(rec["message"]).paper_id`.
- **Tolerant reader (additive vs breaking):** `parse()` distinguishes change classes
  by how strictly it reacts — an **additive** unknown field is **logged + ignored**
  (a new producer never breaks an un-upgraded consumer, so additive needs no
  synchronized deploy); a **subtractive/modified** message (missing required field,
  wrong type, unrecognized `v`) raises `ValidationError`. This is why the sync
  burden (drift tests, expand-contract) only applies to breaking changes.
- **Two physical copies** (the `.json` file + the SQL literal in
  `seed_message_schema()`) because SQL migrations can't read files. **Add a drift
  guard test** (local stack): assert `select public.seed_message_schema()` equals
  `contracts.SCHEMA`. A second guard: `regenerate-contracts.sh` + `git diff --exit-code`
  proves the committed model matches the schema.

### `settings.py`
Add `corpus: str | None`, `doi: str | None`, `source: str | None`, and
`harvest_db_dsn: str | None` (full DSN — local stack for dev, staging pooler for
real). `cli_parse_args=True` already exposes these as `--corpus` / `--source` /
`--arxiv_id` / `--doi`.

### `db.py` (new)
Thin psycopg helpers; `autocommit=False`, `dict_row`.
- `QUEUE_FOR = {"arxiv": "harvest_arxiv", "openalex": "harvest_openalex", "s2": "harvest_s2"}`
- `MAX_RECEIVES = 4` (read_ct cap → terminal error)
- `RATE_INTERVAL = {"arxiv": 3.0, "s2": 1.0, "openalex": 0.1}` (in-process throttle seconds)
- **Seeder:** `submit(conn, *, corpus, arxiv_id=None, doi=None, force=False) -> int | None`
  — insert into `seeds` (`corpus` required, `on conflict (corpus, <id>) do nothing`),
  `select public.publish_seed(paper_id)`; returns the bigint `paper_id` or `None`.
- **Worker:** `read_one(conn, queue, vt=120)`, `archive(conn, queue, msg_id)`,
  `get_identity(conn, paper_id)` (reads `arxiv_id`/`doi` from `seeds`).
- **Upserts** (terminal status in one statement, idempotent
  `on conflict (paper_id) do update`, columns per `20260711011845`):
  - `upsert_source_arxiv(conn, paper_id, ap, status="succeeded", error=None)`
    → sets typed columns + `status` + `fetched_at=now()` (on success).
  - `upsert_source_semantic_scholar(conn, paper_id, *, status="succeeded", error=None, …)`.
  - `mark_source_not_found(conn, paper_id, source)` / `mark_source_error(conn, paper_id, source, error)`
    → upsert a row with just `status` (+ `error`), data columns left null.
- **No** `take_token` / `acquire_token` (deferred).

### `worker.py` (new) — `harvest-work --source X`
Single-threaded drain loop from `harvest-worker-scaling.md`:
```
validate --source ∈ QUEUE_FOR
handler = HARVESTERS[source]          # {"arxiv": harvest_arxiv, "s2": harvest_s2}
loop:
  rec = read_one(queue, vt=120); if None: log "empty", exit 0
  paper_id = contracts.parse(rec["message"]).paper_id     # validate against the v1 contract
  if read_ct > MAX_RECEIVES: mark_source_error(...,"max receives"); archive; continue   # poison
  try:
     ident = get_identity(paper_id)
     handler(conn, paper_id, ident, msg_id)   # fetch + upsert source_X (succeeded/not_found)
     sleep(RATE_INTERVAL[source])             # in-process throttle
     commit; archive; commit                  # terminal outcomes only
  except Exception:
     rollback                                 # transient → left unacked → VT redelivers next run
```
Handlers `harvest_arxiv` / `harvest_s2`: fetch via the source client, upsert
`source_X` to `succeeded` (data) or `not_found`; unexpected errors propagate to the
loop's `except` (transient). Outcome semantics are the table in
`harvest-worker-scaling.md#the-three-outcomes`.

### `seeder.py` (new) — `harvest-seed`
`submit(conn, corpus=…, arxiv_id=…, doi=…)`; logs the published `paper_id` or
"already known, skipped". Never names a queue.

### `sources/` fixes
- `semantic_scholar.py`: remove the `pprint.pformat(...)` debug line (unimported →
  `NameError` on the happy path).
- `arxiv.py`: remove stray debug logging + the unreachable code after the
  `try/except`; add `get_arxiv_paper(id) -> ArxivPaper | None` (rich record the
  upsert needs); keep `get_paper` returning the canonical `Paper`.

### `pyproject.toml`
- **Add** `psycopg[binary]>=3.2`.
- **Remove** `supabase` (REST client — only the deleted `main()` used it) and
  `pdfplumber` (text extraction, deferred) — *verify nothing else imports them first*.
- **Scripts:** drop `harvest = harvest:main`; add
  `harvest-seed = "harvest.seeder:main"`, `harvest-work = "harvest.worker:main"`.
- `uv lock` (CI `lock-check` fails on drift).

### `Dockerfile` / `docker-compose.yml`
The image `ENTRYPOINT ["harvest"]` no longer exists. Compose overrides per service:
`entrypoint: ["harvest-work"]`, `command: ["--source", "<x>"]`. Services: `arxiv`,
`s2` (add `openalex` with the client), `restart: "no"`, env from `.env`
(`HARVEST_DB_DSN`, `MALLOC_ARENA_MAX=2`, `PYTHONUNBUFFERED=1`).

---

## Environments

### Test — pre-merge, ephemeral, free (local + CI per-PR)
`supabase start` + `supabase db reset` builds the full schema from zero on a
throwaway Postgres. Two tests run against it:

| Test | Source client | Network | Where |
|---|---|---|---|
| `e2e_fanout.py` (exists) | — (pure DB) | no | CI per-PR + local — producer fan-out |
| `e2e_worker.py` (**new**, stubbed) | fake `fetch` → canned record | no | CI per-PR + local — drain/archive/**poison**/error plumbing, deterministic |
| `e2e_worker.py` (live) | real `ArxivSource`/`SemanticScholarSource` | yes | local only, behind `RUN_INTEGRATION=1` |

Slots into the `validate` job in `.github/workflows/db-migrations.yml` (right after
`e2e_fanout.py`). Loopback guard keeps it off any real host.

### Staging — post-merge, real, on Supabase
`apply` job runs `supabase db push` to project `vumvzcugxbnkfnujpjjc` on merge.
Workers run as a `docker compose` stack (on your desktop / a VM), pointed at the
staging **session pooler :5432** (6543 forbids prepared statements). Seeder run
manually now (`harvest-seed --corpus X --arxiv_id Y`), promoted to cron, then KEDA.

| | Test | Staging |
|---|---|---|
| Postgres | ephemeral local stack | persistent Supabase project |
| Built from | `db reset` (from zero) | `db push` on merge |
| Workers | stubbed (CI) / live-local | real containers, drain-then-exit |
| DB conn | `127.0.0.1:54322` | staging pooler :5432 |

---

## Test plan

1. **Local dev loop (B):** `HARVEST_DB_DSN` → local stack; `harvest-seed --corpus
   graph-layout --arxiv_id 1706.03762`; `harvest-work --source arxiv` then `--source
   s2`. Assert `source_arxiv` / `source_semantic_scholar` rows `status='succeeded'`
   + each `queue_visible_length` back to 0.
2. **Crash test:** kill a worker between the source upsert and `archive`; re-run →
   converges (idempotent upsert, no dup, queue drains).
3. **Poison test** (via `e2e_worker.py` stubbed): a handler that always raises →
   redelivered until `read_ct > MAX_RECEIVES` → terminal `source_X.status='error'`,
   archived, loop breaks.
4. **CI:** PR touching `v3/supabase/migrations/` → `validate` green (fanout + stubbed
   worker on the local stack).

---

## Risks / open items

- **`requires-python = ">=3.14"` + `psycopg[binary]`** — binary wheels may not exist
  for 3.14 yet. Verify at implementation; fallback is a source build (needs `libpq`)
  or pinning the interpreter. Flagged, not yet resolved.
- **`harvest-implementation.md` is stale** (uuid/`papers`, statusless `source_arxiv`).
  Not touched by this build; noted so implementers use the migration + worker-scaling
  doc as truth. Consider a follow-up to reconcile or retire it.
- **`pg_jsonschema` must be available** on both the staging project and the local
  `supabase` stack (it ships in Supabase's Postgres image and is a supported
  extension, so `create extension if not exists pg_jsonschema` should work in both).
  Verify on `db reset`; fallback is a hand-rolled plpgsql key/type check in
  `publish_seed` instead of `jsonb_matches_schema`.
- **Contract schema is duplicated** (`.json` file + SQL literal). Guard with the
  drift test above so they can't diverge silently.
- **`.json` packaging** — confirm `uv_build` includes `contracts/*.json` in the
  wheel so `importlib.resources` finds it at runtime (verify with `uv build`; add
  build-backend data config if not).

## Deferred (not this build)
`rate_budget`/`take_token`, `harvester_worker`/`harvester_keda_ro` roles, the
openalex worker + `OpenAlexSource`, KEDA/k8s packaging, cron discovery, text
extraction, self-hosted CloudNativePG migration.
