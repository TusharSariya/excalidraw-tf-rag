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

each worker: read_batch(n, vt) ─▶ contracts.parse ─▶ batch-fetch source ─▶ upsert source_X ─▶ archive
             … repeat until the queue is empty, then exit 0  (drain-then-exit)
```

Each drain reads a **batch** of up to `batch_size` messages (per source: arxiv
100, openalex 100, s2 500), waiting up to `WAIT_SECONDS` for a full batch to
accumulate, then batch-fetches them in one source call. Visible-length polling
is **0.1s for the first 1s**, then **1s**; **10s of continuous empty** ends the
wait early so the Job exits (KEDA re-spawns when the queue fills again). Each
seed contributes a single fetch id — the first present identifier in the
source's `identify` order (e.g. arXiv before DOI) — so a batch never exceeds a
source's per-request id cap.

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
| `unpaywall_email` | `UNPAYWALL_EMAIL` | unpaywall — **required** |
| `crossref_mailto` | `CROSSREF_MAILTO` | crossref — optional (polite pool) |
| `core_api_key` | `CORE_API_KEY` | core — **required** |
| `source` | `--source` | `harvest-work` |
| `corpus` / `arxiv_id` / `doi` | `--corpus` / `--arxiv_id` / `--doi` | `harvest-seed` |

Keep secrets in a gitignored `.env`; never commit the `service_role` key.

### Sources and their identifiers

Each `PaperSource` takes the identifier **that source can actually resolve** — they
are not interchangeable:

| Source | `identifier` | Citation graph |
|---|---|---|
| `ArxivSource` | arXiv id (`1706.03762`) | none — arXiv has no citation data, so `get_references`/`get_citations` return `[]` by design |
| `SemanticScholarSource` | arXiv id, mapped to `ARXIV:<id>` | yes — subject metadata + paginated `/citations` and `/references` (`limit=1000`); `s2_writes` upserts ≤1k edges per Write (embedded batch `references` is often empty despite `referenceCount`) |
| `OpenAlexSource` | **DOI or OpenAlex id — never an arXiv id** | yes |
| `UnpaywallSource` | DOI | none — OA-location metadata only (`best_oa_location`) |
| `CrossrefSource` | DOI (batched via `/works?filter=doi:…`) | references only (`crossref_reference`) |
| `DblpSource` | `dblp_id` (DBLP record key) | none |
| `OpenReviewSource` | `openreview_id` (forum id) | none |
| `AclAnthologySource` | `anthology_id` (e.g. `N19-1423`) | none |
| `OpenCitationsSource` | DOI (`doi:` prefixed) | yes — `opencitations_edge`, capped 2000/direction |
| `DataCiteSource` | DOI (`/dois/{doi}`) | none |
| `CoreSource` | DOI | none — OA full-text URL/id |
| `InternetArchiveSource` | `ia_identifier` | none — item/PDF metadata only |
| `ZenodoSource` / `FigshareSource` / `OsfSource` | DOI | none — deposit file + metadata |

The DOI-keyed sources (unpaywall, crossref, opencitations, datacite, core,
zenodo, figshare, osf) only fan out to a seed that carries a `doi`; `dblp_id` /
`openreview_id` / `anthology_id` / `ia_identifier` are the specialised seed
columns the remaining sources key on. Batching: crossref uses the native
DOI-filter batch endpoint; sources without one loop serially but keep the
`get_papers(list) -> dict` contract. See `docs/architecture/harvest-dispatch.md`.

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

### Monitoring on the cluster

The KEDA ScaledJobs run in the `harvester` namespace. Workers are `drain-then-exit`,
so pods live only a few seconds — start a stream *before* seeding, or read from Job
history afterwards.

```bash
# live overview: ScaledJobs (ACTIVE flips when the queue has messages) + Jobs + Pods
watch -n2 kubectl get scaledjobs,jobs,pods -n harvester

# live logs across every harvester pod, incl. ones created after you attach
stern -n harvester harvester

# after the fact (pods are gone; reads the Job's log file directly, no -f)
kubectl get jobs -n harvester --sort-by=.metadata.creationTimestamp | tail
kubectl logs -n harvester job/<job-name>
```

Notes:
- `kubectl get ... -w` accepts only one resource type; `watch` re-running a plain
  `get` is the way to see all three at once.
- If `stern`/`-f` errors with `too many open files`, that's the node's inotify limit
  (`fs.inotify.max_user_instances`), not the worker — non-follow `kubectl logs` still
  works. The ScaledJob/image is defined in `aws-account-infra` (the `cluster_harvester`
  Ansible role), not here.

## Status / history

- **Built:** … plus batched drain (0.10.0) and empty-queue exit / arXiv Client
  pacing (0.11.0). OpenAlex/S2/arXiv persist handlers are wired; OpenAlex
  + S2 proven on the cluster with `0.10.0`; arXiv was blocked by API 429 until
  the 0.11.0 Client pacing fix. Sustained 429s ride on pgmq redelivery (spaced by
  `VT_SECONDS`) rather than an in-process backoff, so a fetch always returns well
  inside the visibility timeout.
- **Next:** re-verify arXiv end-to-end on the cluster after `0.11.0` lands in
  prod ECR; KEDA/compose packaging polish.
- **Deferred:** shared `rate_budget` token bucket (needed once a source scales past
  one pod), least-privilege DB roles, text extraction.
