# harvest

A decoupled, per-source academic-paper harvester. A **seed** (a paper identity in
a corpus) fans out to **one queue per source**; a single-threaded **worker**
drains each queue, fetches that source's metadata, and persists it. Harvest only
(discover → fetch → persist) — no text extraction.

Design docs (canonical): `docs/architecture/harvest-worker-scaling.md` (worker /
scaling semantics) and `docs/architecture/harvest-consumer-build.md` (this build).

## The shape

```
harvest-seed --corpus X --arxiv_id Y
     └─ insert seeds ─ publish_seed(paper_id) ─┬─▶ harvest_arxiv    ─▶ harvest-work --source arxiv
        (validates + stamps the v1 envelope)   ├─▶ harvest_openalex ─▶ harvest-work --source openalex
                                               └─▶ harvest_s2       ─▶ harvest-work --source s2

each worker: read(vt) ─▶ contracts.parse ─▶ fetch source ─▶ upsert source_X ─▶ archive
             … repeat until the queue is empty, then exit 0  (drain-then-exit)
```

- **Producer:** `public.publish_seed(paper_id)` fans a seed out to every active
  queue in `queue_subscriptions`. It stamps + `pg_jsonschema`-validates the
  message before enqueuing, so a malformed payload never ships.
- **Consumer:** `harvest-work --source <arxiv|openalex|s2>` — one container per
  queue, a single-threaded pull loop, `drain-then-exit`.
- **Transport:** messages are read/archived through the Supabase `pgmq_public`
  wrappers over the REST client (see "Prerequisites").

## Message contract

The queue payload is codified once, portably, in `src/harvest/contracts/`:

- `seed_message.v1.json` — the canonical, versioned JSON Schema. v1 is the thin
  envelope `{"v": 1, "paper_id": <bigint>}`.
- `_seed_message_v1.py` — the `SeedMessage` pydantic model, **generated** from the
  schema (`scripts/regenerate-contracts.sh`; do not hand-edit).
- Enforced on both ends: the DB validates in `publish_seed`; the worker parses via
  `contracts.parse()`.
- **Tolerant reader:** an unknown extra field is logged + ignored (additive change
  never breaks an un-upgraded consumer); a missing/mistyped field or an
  unrecognized `v` raises. Bump `v` for breaking changes.

## Configuration

`src/harvest/settings.py` — a shared base plus per-entrypoint settings, resolved
from env / `.env` / CLI flags:

| Setting | Env | Used by |
|---|---|---|
| `supabase_url` | `SUPABASE_URL` | all |
| `supabase_service_key` | `SUPABASE_SERVICE_KEY` (service_role) | all |
| `openalex_api_key` | `OPENALEX_API_KEY` | openalex — **optional**, but 10x the quota (see below) |
| `source` | `--source` | `harvest-work` |
| `corpus` / `arxiv_id` / `doi` | `--corpus` / `--arxiv_id` / `--doi` | `harvest-seed` |

Keep secrets in a gitignored `.env`; never commit the `service_role` key.

### Sources and their identifiers

Each `PaperSource` takes the identifier **that source can actually resolve** — they
are not interchangeable:

| Source | `identifier` | Citation graph |
|---|---|---|
| `ArxivSource` | arXiv id (`1706.03762`) | none — arXiv has no citation data, so `get_references`/`get_citations` return `[]` by design |
| `SemanticScholarSource` | arXiv id, mapped to `ARXIV:<id>` | yes |
| `OpenAlexSource` | **DOI or OpenAlex id — never an arXiv id** | yes |

`OpenAlexSource` is the odd one out because OpenAlex has no arXiv filter: there is no
`ids.arxiv`, and arXiv's own DOI (`10.48550/arXiv.<id>`) 404s. Title search is not a
safe fallback — searching *"Attention Is All You Need"* returns a `type: preprint`
dated 2025 whose PDF is hosted on a scraping mirror. Rather than upsert a
confident-looking wrong record, the source returns `None` for anything it cannot
resolve. A seed carrying only an `arxiv_id` therefore needs a DOI resolved first;
Semantic Scholar's `externalIds` is one route.

The OpenAlex API key is optional — the API answers without one — but it lifts the free
allowance from roughly $0.10/day to $1/day (about 1k → 10k list/filter calls). Get one
free at `openalex.org/settings/api`.

## Running

### Worker (`harvest-work`)

```bash
export SUPABASE_URL=https://<project-ref>.supabase.co
export SUPABASE_SERVICE_KEY='<service_role secret>'   # Dashboard → Settings → API
uv run harvest-work --source arxiv        # drains harvest_arxiv, then exits
```

In Kubernetes each source is one KEDA `ScaledJob` running the same command with a
different `--source`; in Docker, one container per queue
(`ENTRYPOINT ["harvest-work"]`, pass `--source <…>`).

### Prerequisites

The worker reads via the Supabase **Queues** wrappers (`pgmq_public`). Enable them
on the project (Dashboard → Integrations → Queues → "Expose Queues via PostgREST").
Note: that toggle is per-project dashboard state — it is **not** present on the
local `supabase start` stack or in CI.

## Logging

Logs stream to stdout (`PYTHONUNBUFFERED=1`). Loggers are namespaced under
`harvest` (`harvest.worker`, `harvest.arxiv`, `harvest.openalex`,
`harvest.semantic_scholar`), so a run can be filtered by component:

```bash
uv run harvest-work --source arxiv 2>&1 | grep harvest.worker
```

## Status / history

- **Built:** the producer fan-out (`seeds` → `publish_seed` → 3 queues), the
  versioned + DB-enforced message contract, per-entrypoint settings, and the
  `harvest-work` drain loop — proven end-to-end against staging (seed a message,
  read → parse → archive → drain to 0).
- **Next:** the per-source **handler** (fetch via `sources/` + upsert `source_X`
  with `status`, plus the `read_ct` poison cap); the **seeder** CLI
  (`harvest-seed`); KEDA/compose packaging. All three **source clients** now exist
  (`ArxivSource`, `OpenAlexSource`, `SemanticScholarSource`).
- **Deferred:** shared `rate_budget` token bucket (needed once a source scales past
  one pod), least-privilege DB roles, text extraction.
