# RFC: Rearchitecting the RAG/Research Pipeline as a Distributed System

**Status:** draft / thought experiment — written 2026-07-02, the night the GLR re-ingest
failed four different ways (lost futures, checkpoint poisoning, silent consumer death,
queue-deadline storm). Every design decision below is traceable to one of those failures.
**Superseded-by/aligned-with the executed plan (2026-07-03):** the platform-first build
in `pipeline-v2/` and `deploy/` settled several of this RFC's open questions and changed
some designs on contact with real brokers/containers — see
`docs/rearchitecture-v2-decision-log.md` for the authoritative decision record. §1's
failure ledger is historical record and stays as originally written; §2–§5 below have
been updated in place to match what was actually decided and built. Kept for the
narrative "why" — the decision log is the source of truth for "what."

**Goal:** treat harvest → enrich → extract → chunk → embed → index → query as a set of
independently deployable, independently failing services with a control plane, durable
queues, and rollback-capable storage — so that adding a computer, swapping a GPU, or
killing any single worker is an operational non-event instead of an evening.

---

## 1. Why — the failure ledger this design answers

| Incident (2026-07-02) | Root cause | Architectural answer |
| --- | --- | --- |
| 2,266 docs silently gutted to 1-chunk stubs, discovered only by a benchmark regression | Fallback outcome recorded with the same "done" stamp as success; retries skipped it as "unchanged" | Explicit terminal states + dead-letter queue; degraded ≠ complete |
| Ingest wedged at 99.7%, three times | Lost future in a process pool; parent waits forever; no per-task deadline | Broker redelivery: unacked message re-queues after visibility timeout — deadlines are the queue's job, not the worker's |
| CUDA OOM killed the embed consumer thread; extraction kept running into a dead queue | Extraction and embedding co-resident in one process, one failure domain | Separate services; a dead embedder pod restarts; extraction never notices |
| 6 docling workers × CUDA models starved the embedder (OOM storm) | No single owner of VRAM; every process grabs the GPU it sees | All GPU access through one inference service with an admission-controlled VRAM budget (gpu-gateway, promoted) |
| Repair required grepping four log files to reconstruct which docs were damaged | Work state inferred from logs, not stored | Work state is a first-class table/stream, queryable |
| "Rollback" = restoring `.bak` files I happened to make | No versioned storage | Content-addressed artifacts + LanceDB native versioning + object-store snapshots |

The unifying theme: **the current pipeline is a single program that happens to have
stages; the target is stages that happen to form a pipeline.**

---

## 2. Target architecture

```
                        ┌─────────────────────────────────────────────┐
                        │  Control plane (k8s API + Argo Workflows)    │
                        └─────────────────────────────────────────────┘
                                           │
   ┌───────────┐   topics   ┌──────────────┼──────────────────────────────┐
   │  Harvest  │──────────▶│               ▼                               │
   │  (CPU)    │  docs.new  │   Apache Kafka 4.x, KRaft + share groups    │
   └───────────┘            │ (KIP-932) — durable log, explicit per-msg   │
   ┌───────────┐            │ ack/reject/release, app-level DLQ           │
   │  Enrich   │◀──────────▶│ docs.new → docs.enriched → chunks.ready →   │
   │  (CPU)    │            │ chunks.embedded → index.committed           │
   └───────────┘            │  + docs.deadletter.<stage> (app-produced)   │
   ┌───────────┐            └──────────────┬──────────────────────────────┘
   │  Extract  │  consumes docs.enriched   │   ┌──────────────────────────┐
   │  docling  │───────────────────────────┘   │  Inference service        │
   │  (CPU/GPU)│  emits chunks.ready           │  (gpu-gateway, promoted)  │
   └───────────┘───────────/v1/embeddings ────▶│  owns ALL VRAM;           │
   ┌───────────┐  consumes chunks.ready        │  bin-packed co-residency, │
   │  Embed    │───────────────────────────────▶  admission control,      │
   │  (thin)   │  emits chunks.embedded        │  LRU eviction             │
   └───────────┘                               └──────────────────────────┘
   ┌───────────┐  consumes chunks.embedded     ┌──────────────────────────┐
   │  Indexer  │──────────────────────────────▶│  Storage: SeaweedFS (S3) │
   │ (single-  │  emits index.committed        │  + LanceDB (versioned) + │
   │  writer)  │                               │  tantivy + CloudNativePG │
   └───────────┘                               │  (work-state + manifest, │
   ┌───────────┐                               │  Barman Cloud backups)   │
   │  Query API│◀── reads index + inference    └──────────────────────────┘
   └───────────┘    service (rerank/HyDE/synthesis)
```

Decided (see decision log): **Apache Kafka 4.x** over Redpanda (Apache 2.0 vs BSL;
share groups GA gives native per-message ack without a proprietary fork), **SeaweedFS**
over MinIO (MinIO's repo was archived April 2026), **Argo Workflows** over Temporal for
the control plane (k8s-native, YAML DAGs; Temporal kept in reserve), **CloudNativePG**
with the Barman Cloud plugin for Postgres (declarative PITR; the in-tree
`barmanObjectStore` path is deprecated/removed upstream).

### 2.1 Services (each = one container image, one failure domain)

| Service | Replaces | Scaling axis | GPU? |
| --- | --- | --- | --- |
| `harvest` | `harvest/*.py` CLI runs | per-source rate limits, not CPU | no |
| `enrich` | `harvest/enrich.py`, citation matching | API quota | no |
| `extract` | ProcessPoolExecutor docling workers | **CPU cores across all nodes** — the actual bottleneck tonight | optional (docling layout models via inference service) |
| `embed` | in-process consumer thread | thin: batches chunks, calls inference service | via service |
| `indexer` | LanceDB/tantivy writes in ingest | single-writer per index (serialize commits) | no |
| `inference` | `gpu-gateway` (already exists — this is the one component that survives unchanged) | one per GPU; add GPUs = add replicas | **yes — sole owner** |
| `query-api` | `query/` CLI surfaces | stateless, horizontal | via service |
| `eval-runner` | `eval benchmark`, map-gate | batch jobs (Argo CronWorkflow) | via service |

Key inversion: **workers never touch the GPU directly.** Extraction pods that want
docling's layout model call the inference service, same as embedding does. Tonight's OOM
storm becomes impossible by construction — VRAM has exactly one owner with admission
control. This is why `gpu-gateway` is the seed of the whole design: it already implements
bin-packed co-residency and atomic LRU eviction for one 8 GB card; promoting it to "the
only door to any GPU" is the architecture.

### 2.2 Queues and the dead-letter discipline

**Apache Kafka 4.x, KRaft mode, KIP-932 share groups** (GA since 4.2.0, default-on),
one topic per stage boundary. The share-group spike against a real Kafka 4.3.1 broker
plus `confluent-kafka-python`'s `ShareConsumer` (pinned 2.15.0) changed several details
from the original draft:

- A message is acked (or explicitly rejected/released) only when the stage produced its
  *successful* artifact — the acking model survives from the draft unchanged.
- **No automatic broker DLQ.** This was the RFC's original assumption and it's wrong for
  share groups: Kafka silently archives exhausted records (`delivery_count` past the
  broker limit) with no dead-letter topic of its own. The dead-letter discipline has to
  be implemented in the app: `StageConsumer` tracks `delivery_count` and, at an
  app-level threshold (4, below the broker's own archive limit so the app always gets
  there first), produces to `docs.deadletter.<stage>` itself, then ACKs the original
  message and writes a `failed` work-state row. Fallback outcomes
  (`empty_pdf`, `extraction_timeout`, `worker_error`) still never flow downstream
  stamped as complete — that rule, the one that prevents the 2,266-doc incident, is
  unchanged; it's just enforced by application code against the work-state ledger
  (`pipeline_v2.workstate`, `state IN ('ok','degraded','failed')`, `indexable_work_state`
  view only ever surfaces `ok`) rather than by broker behavior.
- Redelivery-on-no-ack replaces tonight's hand-rolled per-future deadlines: a worker that
  wedges or dies simply never acks; the broker re-delivers to another worker after the
  record lock expires (crash → redelivery verified at ~6s against a real broker, with
  `delivery_count` incrementing). Max-attempts exceeded → app-level DLQ as above, never
  an infinite loop.
- **The 60s lock-clamp finding:** `group.share.max.record.lock.duration.ms` defaults to
  60s and silently clamps any longer per-message lock a client requests — a 600s lock
  request for a 300s-worst-case PDF extraction would have failed silently without
  raising this broker-side max. Provisioning now sets it to 900s explicitly.
- **`ShareConsumer` is officially Preview** (targeted GA ~2H2026) — pinned at
  `confluent-kafka==2.15.0` behind a `WorkQueue` adapter interface with a
  `ClassicGroupQueue` (plain consumer groups + Postgres claim rows) fallback, so the
  fallback path is exercised in tests (parity-tested), not just theorized.
- Every message carries `{doc_id, content_sha256, stage_fingerprint}` so redelivery is
  idempotent (see 2.4); duplicate-redelivery idempotency is covered by the P1/P2
  integration test suite.

### 2.3 Kubernetes layout (vcluster)

- **Host cluster:** k3s on the GPU box (later: + every machine you add). NVIDIA
  device plugin + node feature discovery labels GPU nodes (`gpu.model=3060ti`,
  `gpu.vram=8Gi`).
- **vclusters** give you the isolation you want without N physical clusters:
  - `vc-pipeline` — the services above.
  - `vc-eval` — eval runs, resource-capped so a benchmark can never starve ingest.
  - `vc-dev` — scratch experiments against a *restored snapshot* of prod indexes.
  - GPU sharing across vclusters is mediated by the inference service, not by the
    scheduler — consumer cards have no MIG, so k8s-level GPU splitting is a dead end;
    HTTP-level admission control is the honest mechanism. The inference service runs in
    the host cluster (or `vc-pipeline`) and is exposed to all vclusters as a Service.
- **Workflow control plane:** Argo Workflows (or Temporal if you want code-first) for
  the DAG-shaped jobs: "harvest source X → enrich → fan-out extract → gate on eval."
  Cron for scheduled harvests. This is your dashboard for "what is the pipeline doing
  right now and what failed," replacing tonight's `grep reingest*.log`.
- **Portability payoff:** switch computers = `kubectl drain` + join the new node; add a
  GPU = deploy another inference replica with a node selector; everything else finds it
  via the Service DNS name. `RAG_GPU_GATEWAY_URL` already proved this pattern works —
  the code path exists today.

### 2.4 Storage — content-addressed, versioned, rollback-first

Three tiers, replacing "files in `data/` + hand-made `.bak`s":

1. **Object store (SeaweedFS, single-node, no replication):** raw PDFs, page-cache
   artifacts, chunk artifacts — keyed by `sha256(content) + stage fingerprint` (the
   current page/chunk/embed caches are already content-addressed; this lifts them into a
   shared, multi-node store). Versioning on; artifacts are immutable, so "rollback" is
   just re-pointing. MinIO was the original choice but its repo was archived April 2026;
   SeaweedFS was picked over Garage (AGPL licensing) and RustFS (still alpha).
2. **LanceDB over object store, native versioning:** Lance tables are already
   append + version-manifest based — `checkout(version)` is built in. Every indexer
   commit is tagged with the ingest run id. Rolling back tonight's gutted index would
   have been `table.checkout_version(pre_run)` instead of a 5-hour re-extraction.
   tantivy snapshots ride along as tarballs in SeaweedFS keyed by the same run id.
3. **CloudNativePG** (not a plain StatefulSet): manifest + work-state (per-doc: stage,
   attempts, terminal state, reason, artifact pointers) + harvest ledger, with the
   Barman Cloud plugin doing declarative PITR (the in-tree `barmanObjectStore` path is
   deprecated/removed upstream, so the plugin is required, not optional). The manifest
   stops being a 6 MB JSON file with merge-clobber bugs; DOI-collision guards become
   uniqueness constraints and the title-similarity check becomes a trigger/
   application-level gate with an audit row. P0b's acceptance gate is an off-box
   restore: CNPG PITR restored on a non-desktop machine, not just a same-box smoke test.

Filesystem choice becomes almost irrelevant (the rollback story lives in SeaweedFS
versioning + Lance versions + CNPG PITR), but btrfs/ZFS on the nodes is still nice
for fast local scratch snapshots. Backup = `restic`/`rclone` of SeaweedFS + Barman Cloud
for Postgres, replacing `backup-manifest.toml`'s file list.

### 2.5 Observability (the part that found tonight's bugs by accident)

**Grafana LGTM** (Loki + Tempo + plain Prometheus, not Mimir) — chosen over OpenObserve
for the industry-standard stack, OTel SDK instrumentation either way. Mimir was
evaluated and dropped: it has no single-binary mode, which blows the 8 GB VRAM/host
budget this box runs under; Loki and Tempo do have single-binary modes and are kept.

- Per-stage Prometheus metrics: queue depth, ack rate, app-level DLQ depth, extraction
  seconds histogram, VRAM headroom, tokens/batch. Tonight's "queue saturates while GPU
  idles" and "2,266 fallbacks" would both be red graphs within minutes, not grep
  archaeology.
- Alert rules: DLQ depth > N, stage throughput = 0 for > 10 min (the wedge detector),
  inference service admission rejections (the OOM precursor).
- One Grafana dashboard = the sit-rep.

---

## 3. What survives from the current codebase

| Keep as-is | Becomes |
| --- | --- |
| `gpu-gateway` scheduler (bin-packing, LRU eviction) | the inference service core — this is the crown jewel |
| page/chunk/embed cache keying (content sha + fingerprint) | SeaweedFS object keys |
| chunking, reference-stripping, docling wrapper | `extract` service internals, unchanged logic |
| embed profiles TOML | inference service model registry |
| eval harness, qrels, regression floor | `eval-runner` batch jobs |
| harvest tools + ledger | `harvest`/`enrich` services; ledger schema → Postgres |
| manifest merge guards (title similarity, sha dedup) | Postgres constraints + application gate |
| `rag_common.local_llm` transports | thin clients of the inference service |

The domain logic barely changes; what changes is that the *process boundaries* move to
the failure boundaries.

---

## 4. Migration path — executed as platform-first, not the original phase-0..5 order

The original draft below proposed a failure-first ordering (dead-letter semantics,
then storage, then cluster, then queues). The user decided **platform-first** instead
(decision log D3): standing up the platform (k8s, brokers, object store, Postgres) is
itself part of the learning goal here, not just a means to an end, so it isn't deferred
to the back half. Risk from building platform before proving failure semantics is
mitigated by crash-injection gates at every phase and a standing **3-week stall
kill-criterion** — if a phase stalls that long without a shippable result, fall back to
the failure-first order instead.

Executed phasing:

- **P0a — GPU boundary.** gpu-gateway containerized (pinned pytorch/CUDA base image,
  non-root, HEALTHCHECK) and made the only GPU tenant; token-aware batching +
  OOM-catch-halve-retry lands here (generalizes the old task #13 beyond one tool).
- **P0b — platform services.** k3s host cluster, OpenTofu platform layer + a layer-0
  bootstrap script (installs k3s/kubectl/helm/tofu without root where possible),
  kustomize for service manifests; CloudNativePG + Barman Cloud plugin; SeaweedFS.
  Acceptance gate: an **off-box CNPG PITR restore**, not just a same-box smoke test.
- **P1 — work-state ledger + manifest migration.** `pipeline-v2` package: Postgres-backed
  per-(doc, corpus, stage) outcome ledger with a hard `CHECK (state IN ('ok','degraded',
  'failed'))` and an `indexable_work_state` view that only ever surfaces `ok` rows —
  this closes the 2,266-doc bug class structurally, not by convention. Manifest
  migration imports v1 `manifest.json` with a byte-level round-trip verification.
- **P2 — messaging backbone.** Kafka 4.x share groups (`WorkQueue` interface: share-group
  + classic-group fallback), app-level DLQ, pydantic message contracts with exported
  JSON Schemas, a generic `StageConsumer` (poison→DLQ, idempotency skip, ok-only
  propagation). Preceded by a share-group Python-client spike against a real broker
  (decision log lane A4) before committing to the design.
- **P3 — service split.** extract/embed/indexer become independent consumers behind
  the messaging backbone from P2; the ProcessPoolExecutor, `_drain_one`, and all its
  deadline code get deleted rather than debugged.
- **P4 — control plane + twin collapse.** Argo Workflows for harvest→eval DAGs;
  GLR/RLR plugin-module collapse happens here, resequenced from earlier in the plan and
  gated on both corpora passing their eval floors unchanged first.
- **P5 — second machine / second GPU.** Join node, add an inference replica, raise
  extract replica count — the payoff the whole design buys.

Every phase carries **crash-injection gates**, not just functional tests: kill -9 mid
extract/embed/index, kill the indexer during every rollback step (old run must stay
queryable), kill workers under redelivery to prove idempotency, and a SeaweedFS storage
torture gate (kill -9 mid-write, disk-full, restic restore).

**v2-coexistence (standing rule):** `pipeline-v2` is built *alongside* the existing v1
tools, not as a replacement in place — v1 packages are additive-only during this build,
data planes stay separate, and nothing gets deleted until the user explicitly retires
v1. This RFC's "delete `_drain_one`" language above (P3) still only applies once v2 is
proven, not on the same day it lands.

## 5. Open questions — resolved

Most of the open questions below were live when this RFC was drafted; they're settled
now (decision log is authoritative):

- **Kafka vs Redpanda vs NATS JetStream — decided: Apache Kafka 4.x** (KRaft + KIP-932
  share groups). Redpanda's BSL license was the deciding factor against it (license
  consistency with the rest of the platform-first stack, which is Apache-licensed
  end to end); NATS JetStream was considered lighter-weight but off-thesis for the
  learning goal.
- **Temporal vs Argo — decided: Argo Workflows** (P4), k8s-native YAML DAGs; Temporal
  stays in reserve if pipeline logic later needs to live in replayable Python code.
- **IaC tool — decided: OpenTofu** (platform layer) + a layer-0 bootstrap script for the
  parts that must run before Tofu can reach a cluster, + kustomize for service
  manifests, with Argo CD as an optional P4 addition. Chosen over Terraform (BSL —
  license consistency again), Ansible, Pulumi, kubectl-only, and GitOps-only; existing
  Terraform skills transfer directly to OpenTofu's syntax.
- **Postgres — decided: CloudNativePG**, not a plain StatefulSet or SQLite-then-migrate;
  this is the 2026 consensus operator and gives declarative PITR from the start, so the
  work-state ledger (P1) is Postgres-backed from day one rather than SQLite-first.
- Does `extract` call docling's layout model through the inference service or run
  CPU-only in-pod? Still open — not exercised by the P0a–P2 work executed so far;
  CPU-only in-pod remains the default, service-routed as a later experiment.
- Embedding batch policy moved into the inference service (token-aware batching,
  OOM-catch-halve-retry) — done in P0a, generalizing the old task #13 beyond one tool.
