# Harvest workers & autoscaling

How to implement, run, and scale the **consumer** side of the harvester: the
workers that drain the per-source PGMQ queues, fetch source metadata, and persist
it. Assumes the **producer** side already exists (this repo): `seeds`,
`queue_subscriptions`, `publish_seed(bigint)`, the three `harvest_*` queues, and
the `source_*` outcome tables.

> This doc reflects the **current** schema — `seeds` (bigint identity + required
> `corpus`), dumb catalog, per-source status in `source_*` tables — and the design
> decisions recorded below. Where it disagrees with
> `docs/architecture/harvest-implementation.md` (written against the older
> `papers`/uuid schema), **this doc wins** for the worker/scaling layer.

---

## The model in one picture

```
harvest-seed (corpus, id) ─▶ insert seeds ─▶ publish_seed(paper_id)
                                                  │  (fans out over queue_subscriptions)
                          ┌───────────────┬───────┴───────────────┐
                          ▼               ▼                        ▼
                   harvest_arxiv    harvest_openalex          harvest_s2      ← one PGMQ queue / source
                          │               │                        │
                   ScaledJob arxiv  ScaledJob openalex        ScaledJob s2    ← KEDA, metric = queue_visible_length
                     max=1            max=3                     max=2         ← per-source pod cap
                          ▼               ▼                        ▼
        harvest-work --source X (single-threaded drain loop):
          read(vt) ─▶ acquire_token(X) ─▶ fetch source X ─▶ upsert source_X (status) ─▶ archive
```

- **Concurrency comes from KEDA pod count**, not threads. One worker = one thread.
- **Request rate is governed by a shared Postgres token bucket** (`rate_budget`),
  so pod count and rate are decoupled.
- **The queue is the scaling signal**: KEDA polls `queue_visible_length` and runs
  one Kubernetes Job per visible message, up to the per-source cap, scaling to
  **zero** when the queue drains.

---

## Core decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Threading | **Single-threaded** worker, one thread per container | Crash affects exactly one message; no in-process concurrency to reason about. Your throughput ceiling is the per-source rate limit, not machine concurrency — extra threads just queue on the token bucket. |
| Concurrency knob | **KEDA pod count** (`maxReplicaCount` per source) | One dial. Rate limits are so low (arxiv 1/3s, s2 1/s) that even one pod saturates them; only openalex benefits from a few pods. |
| Rate limiting | Shared `rate_budget` token bucket in Postgres | N pods of one source draw from one bucket, so `maxReplicaCount` (pods) and rate (tokens) stay independent and correct. **Required once any `maxReplicaCount > 1`.** |
| Pod granularity | **Drain loop** — process messages until the queue empties, then exit (not one-message-one-pod) | Amortizes pod startup (~0.5–1s) across many messages. |
| Delivery | At-least-once + **idempotent** upserts | pgmq redelivers after the visibility timeout; re-processing converges because writes are `on conflict do update`. |
| Dead-letter | **No DLQ.** Terminal failures land as `source_<x>.status='error'` | The `source_*` table is a queryable, typed dead-letter store; redrive is a `SELECT` + re-publish. See [Failure semantics](#failure-semantics). |
| Language | **Python** (stay) | Language affects memory baseline & cold start ~5–10×, but is irrelevant to throughput here (rate-limited, network-bound). Absolute footprint is ≤6 pods; API clients are Python-native. |
| Autoscaler | KEDA **`ScaledJob`** per queue (default) | One Job per message, clean scale-to-zero, crash isolation. See [Cold start](#cold-start) for the `ScaledObject` fallback. |
| Worker DB access | `psycopg` → Supabase **session pooler, port 5432** | 6543 (transaction pooler) forbids prepared statements, which psycopg/pgx use. |
| Worker DB identity | Role **`harvester_worker`** (not `service_role`/REST) | This role has pgmq grants, so `publish_seed` (`SECURITY INVOKER`) works for it directly. (The REST/`service_role` path needs `SECURITY DEFINER` — that's the `_test` fan-out test, a different caller.) |

---

## Message & failure semantics

pgmq gives you SQS-style **visibility timeout** and a **receive counter**, but
**no automatic DLQ**. We lean on both and skip the DLQ.

### Visibility timeout (crash-safe redelivery)

`pgmq.read(queue, vt, qty)` makes a message invisible for `vt` seconds. If the
worker dies before `pgmq.archive`, the message reappears after `vt` and a fresh
pod reprocesses it. Extend for long work with `pgmq.set_vt(queue, msg_id, vt)`.
KEDA's `queue_visible_length` **excludes in-flight (invisible) messages**, so a
message being processed never tricks KEDA into spawning a duplicate pod for it.

### The three outcomes

| Outcome | When | Action |
|---|---|---|
| `succeeded` | got the data | upsert `source_X` row with data + `status='succeeded'` + `fetched_at`; `archive` |
| `not_found` | source genuinely doesn't have this paper | upsert `status='not_found'`; `archive`. **Terminal, expected — not an error, never retry.** |
| `error` (transient) | API 5xx/429/timeout | do **nothing** — leave the message unacked so the visibility timeout **redelivers** it. These free retries are the point. |
| `error` (terminal) | `read_ct` exceeds the cap | upsert `status='error'` + `error` text; `archive`. Stops the poison loop. |

### Why no DLQ

Without a `read_ct` cap, a message that always fails redelivers **forever**, and
KEDA spawns a pod each time — an infinite poison loop. The cap converts that into
a terminal `source_X.status='error'` row. That row **is** the dead-letter record —
queryable and typed, better than raw JSON in a side queue. Redrive is trivial:

```sql
-- re-enqueue everything that errored on arxiv (e.g. after a source outage)
select public.publish_seed(paper_id) from public.source_arxiv where status = 'error';
```

If you later want *automated, bounded* redrive, add `error_count` / `last_error_at`
to the `source_*` tables and a scheduled sweep that re-publishes `error` rows older
than N hours, up to M times. Deferred until needed.

---

## Resource sizing (network-bound)

CPU and memory behave oppositely under cgroups: **CPU limits throttle, memory
limits kill (OOM).** Size accordingly.

| Resource | Value | Why |
|---|---|---|
| `requests.cpu` | `50m` | True average is ~0 (blocked on sockets = not scheduled = 0% CPU). Low request keeps bin-packing dense. |
| `limits.cpu` | **unset** (or `1`) | A CPU limit throttles the short bursty work (JSON/psycopg decode) even while node cores sit idle. Never cap a bursty network-bound worker tightly. Watch `container_cpu_cfs_throttled_periods_total`. |
| `requests.memory` | `128Mi` | Steady RSS of a metadata worker (Python + psycopg + requests + pydantic ≈ 100–150 MB). |
| `limits.memory` | `256Mi` | ≥ peak + headroom. Memory can't burst — exceed and you're OOM-killed. |
| env | `MALLOC_ARENA_MAX=2`, `PYTHONUNBUFFERED=1` | Cap glibc arena fragmentation (single-threaded → 2 arenas is plenty); unbuffered logs. |

**QoS = Burstable** (mem request < limit, no cpu limit) — the right class for
bursty network work. Avoid **Guaranteed** (forces a cpu limit → throttling) and
**BestEffort** (first to be killed).

**Measure before trusting the numbers** — run one worker under load and read the
real high-water mark, then set `limits.memory ≈ peak × 1.5–2`:

```bash
cat /sys/fs/cgroup/memory.peak     # cgroup v2, inside the container
kubectl top pod <pod>              # k8s
docker stats --no-stream           # compose
```

Because pods drain-then-**exit**, memory can't creep — sized once, done. (Python
doesn't return freed memory to the OS, but the pod dying frees everything.)

---

## Cold start

Python cold start is ~0.5–1s (interpreter + `pydantic`/`psycopg` imports), paid on
every pod spawn. The **drain loop already amortizes it** across all the messages a
pod processes, so for the `ScaledJob` model it's a small fraction of pod lifetime.

If a source ever needs cold start *gone* (high volume, latency-sensitive), the
answer is **not** CRIU/SnapStart-style snapshotting (privileged, fragile with open
sockets, sub-second win) — it's to **stop being ephemeral**: run that source as a
long-lived KEDA **`ScaledObject` Deployment** that loops on the queue for hours
(startup paid once) and still scales to zero (`minReplicaCount: 0`). Trade: you
lose per-message crash isolation.

Cheaper wins that need no infra:
- **Lazy-import** heavy modules per source (an arxiv pod never imports the S2/pydantic graph).
- **Precompile bytecode** in the image (`python -m compileall`; `uv` caches `.pyc`).
- `python -X importtime` to find and prune slow imports.

**Default: `ScaledJob` for all three sources. Revisit `ScaledObject` only if
cold-start latency measurably bites.**

---

## Schema prerequisites (still to add)

The producer side exists. Two migrations are still needed before workers run,
reconciled to the current schema:

### `rate_budget` + `take_token()` (needed once any `maxReplicaCount > 1`)

```sql
create table public.rate_budget (
    source          text primary key,        -- type key: 'arxiv' | 'openalex' | 's2'
    tokens          numeric not null,
    burst           numeric not null,
    refill_per_sec  numeric not null,
    last_refill     timestamptz not null default now()
);
insert into public.rate_budget (source, tokens, burst, refill_per_sec) values
    ('arxiv',    1,  1,  0.33),               -- ~1 req / 3s
    ('openalex', 10, 10, 10),                 -- polite pool ~10 req/s
    ('s2',       1,  1,  1);                   -- ~1 req/s unauthenticated
alter table public.rate_budget enable row level security;

-- Atomic refill-on-read + take-one. Returns true iff a token was available.
create function public.take_token(p_source text) returns boolean
language plpgsql as $$
declare ok boolean;
begin
    update public.rate_budget
       set tokens = least(burst, tokens + refill_per_sec
                          * extract(epoch from (now() - last_refill))),
           last_refill = now()
     where source = p_source;
    update public.rate_budget
       set tokens = tokens - 1
     where source = p_source and tokens >= 1
    returning true into ok;
    return coalesce(ok, false);
end $$;
```

### `harvester_worker` / `harvester_keda_ro` roles (least privilege)

```sql
-- Worker: catalog + source tables + pgmq + fan-out + rate bucket.
create role harvester_worker login password 'SET_OUT_OF_BAND';
grant usage on schema public, pgmq to harvester_worker;
grant select, insert, update on
    public.seeds, public.source_arxiv, public.source_openalex,
    public.source_semantic_scholar to harvester_worker;
grant select on public.queue_subscriptions to harvester_worker;
grant select, update on public.rate_budget to harvester_worker;
grant execute on function public.publish_seed(bigint) to harvester_worker;  -- bigint now
grant execute on function public.take_token(text)     to harvester_worker;
grant execute on all functions in schema pgmq to harvester_worker;
grant select, insert, update, delete on all tables in schema pgmq to harvester_worker;

-- KEDA scaler: read-only, only needs pgmq.metrics().
create role harvester_keda_ro login password 'SET_OUT_OF_BAND';
grant usage on schema pgmq to harvester_keda_ro;
grant execute on function pgmq.metrics(text) to harvester_keda_ro;
grant select on all tables in schema pgmq to harvester_keda_ro;
```

> Set real passwords out-of-band (Ansible Vault). Because `harvester_worker` holds
> pgmq grants, `publish_seed` as `SECURITY INVOKER` works for it — no
> `SECURITY DEFINER` needed on the real function.

---

## Worker & seeder code (`v3/harvest/`)

Add to `pyproject.toml`: `psycopg[binary]>=3.2`, and two console scripts —
`harvest-seed = "harvest.seeder:main"`, `harvest-work = "harvest.worker:main"` —
then `uv lock`. `settings.py` gains `corpus`, `source`, and a `db_dsn` (session
pooler).

### `db.py` — the key helpers (reconciled to `seeds`/`source_*`)

```python
QUEUE_FOR = {"arxiv": "harvest_arxiv", "openalex": "harvest_openalex", "s2": "harvest_s2"}
MAX_RECEIVES = 4  # read_ct cap → terminal error


def submit(conn, *, corpus, arxiv_id=None, doi=None, force=False) -> int | None:
    """Insert a seed (corpus REQUIRED) and fan out. Returns bigint paper_id or None."""
    col, val = ("arxiv_id", arxiv_id) if arxiv_id else ("doi", doi)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            f"insert into seeds (corpus, {col}) values (%s, %s) "
            f"on conflict (corpus, {col}) do nothing returning paper_id",
            (corpus, val),
        )
        row = cur.fetchone()
        if row is None:
            if not force:
                return None
            cur.execute(f"select paper_id from seeds where corpus=%s and {col}=%s", (corpus, val))
            row = cur.fetchone()
        paper_id = row["paper_id"]  # bigint
        cur.execute("select public.publish_seed(%s)", (paper_id,))
    return paper_id


def read_one(conn, queue, vt=120):
    with conn.cursor() as cur:
        cur.execute("select * from pgmq.read(%s, %s, 1)", (queue, vt))
        return cur.fetchone()  # msg_id, read_ct, ..., message


def archive(conn, queue, msg_id):
    with conn.cursor() as cur:
        cur.execute("select pgmq.archive(%s, %s)", (queue, msg_id))


def acquire_token(conn, source, *, timeout=60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with conn.cursor() as cur:
            cur.execute("select public.take_token(%s) as ok", (source,))
            got = cur.fetchone()["ok"]
        conn.commit()
        if got:
            return True
        time.sleep(0.5)
    return False


def upsert_source_arxiv(conn, paper_id, ap, status="succeeded", error=None):
    """Idempotent upsert; sets status + fetched_at. See migration for full column list."""
    ...  # insert ... on conflict (paper_id) do update set ..., status=%s, error=%s, fetched_at=now()


def mark_source_error(conn, paper_id, source, error):
    """Terminal error: upsert the source_X row to status='error' with the message."""
    ...
```

### `worker.py` — single-threaded drain loop with the read_ct cap

```python
def main():
    s = HarvestSettings()
    source = s.source
    if source not in db.QUEUE_FOR:
        raise SystemExit(f"--source must be one of {sorted(db.QUEUE_FOR)}")
    queue = db.QUEUE_FOR[source]
    handler = HARVESTERS[source]  # {"arxiv": harvest_arxiv, "s2": harvest_s2, ...}

    with db.connect(s.db_dsn) as conn:
        while True:  # drain loop: amortize startup across messages
            rec = db.read_one(conn, queue, vt=120)
            if rec is None:
                log.info("queue %s empty, exiting", queue)
                return
            msg_id, read_ct, paper_id = rec["msg_id"], rec["read_ct"], rec["message"]["paper_id"]

            if read_ct > db.MAX_RECEIVES:  # poison → terminal error, stop retrying
                db.mark_source_error(conn, paper_id, source, "max receives exceeded")
                db.archive(conn, queue, msg_id)
                conn.commit()
                log.warning(
                    "poison msg_id=%s paper_id=%s -> source_%s error", msg_id, paper_id, source
                )
                continue

            try:
                ident = db.get_identity(conn, paper_id)  # arxiv_id/doi from seeds
                handler(conn, paper_id, ident, msg_id)  # writes source_X (succeeded/not_found)
                conn.commit()
                db.archive(conn, queue, msg_id)
                conn.commit()  # terminal outcomes only
            except Exception:
                conn.rollback()  # transient: leave unacked → VT redelivers
                log.exception("transient fail msg_id=%s; will redeliver", msg_id)
```

`get_identity` reads the seed's `arxiv_id`/`doi` (the payload carries only
`paper_id`; `corpus` is already attached to the seed row). Each `handler` acquires
a token, fetches, and upserts `source_X` to `succeeded` or `not_found` — both
terminal (archived). Only unexpected exceptions leave the message for redelivery.

### `seeder.py`

```python
def main():
    s = HarvestSettings()
    with db.connect(s.db_dsn) as conn:
        paper_id = db.submit(conn, corpus=s.corpus, arxiv_id=s.arxiv_id, doi=s.doi)
    log.info("published paper_id=%s" if paper_id else "already known, skipped", paper_id)
```

---

## Usage

### Local (no k8s), against staging or a local stack

```bash
cd v3/harvest && uv sync
# seed one paper into a corpus, fan out to all queues
uv run harvest-seed --corpus graph-layout --arxiv_id 1706.03762
# drain each queue: single-threaded, processes until empty, then exits
uv run harvest-work --source arxiv
uv run harvest-work --source s2
```
Verify: a `seeds` row; `source_arxiv` / `source_semantic_scholar` rows with
`status='succeeded'`; each `queue_visible_length` back to 0. **Crash test:** kill
`harvest-work` between the source upsert and `archive`; re-run and confirm it
converges (no dup — the upsert is idempotent — and the queue drains).

### Kubernetes (KEDA `ScaledJob` per source)

One `ScaledJob` per queue, resource settings from [above](#resource-sizing-network-bound):

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata: { name: harvester-arxiv, namespace: harvester }
spec:
  jobTargetRef:
    backoffLimit: 3
    template:
      spec:
        restartPolicy: Never
        containers:
          - name: harvest-work
            image: <ecr>/harvest:<tag>
            command: ["harvest-work", "--source", "arxiv"]
            env:
              - { name: MALLOC_ARENA_MAX, value: "2" }
            envFrom: [{ secretRef: { name: harvester-secrets } }]   # HARVEST_DB_DSN etc.
            resources:
              requests: { cpu: "50m", memory: "128Mi" }
              limits:   { memory: "256Mi" }                          # note: no cpu limit
  pollingInterval: 15
  maxReplicaCount: 1                 # per-source cap: arxiv 1, openalex 3, s2 2
  scalingStrategy: { strategy: accurate }
  triggers:
    - type: postgresql
      metadata:
        query: "SELECT queue_visible_length FROM pgmq.metrics('harvest_arxiv')"
        targetQueryValue: "1"        # 1 Job per pending message, capped by maxReplicaCount
      authenticationRef: { name: harvester-pg }   # TriggerAuthentication → harvester_keda_ro DSN (5432)
```

Full Ansible/Helm packaging (namespace, secret from Vault, one `ScaledJob` per
source via a loop, discovery `CronJob`) lives in the `cluster_harvester` role —
see `harvest-implementation.md` §"Repo 3" (still valid; just swap its
`max_replicas`/resource numbers for the ones here and point grants at `seeds`).

### Observe & operate

```sql
-- queue depths (what KEDA sees)
select queue_name, queue_length, queue_visible_length
from (values ('harvest_arxiv'),('harvest_openalex'),('harvest_s2')) q(name)
     , lateral pgmq.metrics(q.name);

-- per-source outcomes
select status, count(*) from public.source_arxiv group by status;

-- redrive everything that errored on a source
select public.publish_seed(paper_id) from public.source_arxiv where status='error';
```

---

## Adding a fourth source (the payoff of decoupling)

1. Migration: `select pgmq.create('harvest_<new>');` + `insert into queue_subscriptions …`
   + `insert into rate_budget …` + `create table public.source_<new> (…, status source_status, …)`.
2. Worker: add an `upsert_source_<new>` + a `harvest_<new>` handler; register it in `HARVESTERS`.
3. K8s: append one `ScaledJob` (one entry in `cluster_harvester_sources`).

The **seeder never changes** — `publish_seed()` picks up the new subscription
automatically.

## Deferred (not in this build)

Text extraction, OpenAlex discovery-since-watermark + `OpenAlexSource`, S3 artifact
backup, automated bounded redrive (`error_count` sweep), long-lived `ScaledObject`
workers, title→paper resolution.
