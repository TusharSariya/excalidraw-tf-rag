# Harvester implementation guide

A step-by-step, copy-pasteable guide to build the decoupled, KEDA-autoscaled harvester.
Scope is **harvest only** (discover → fetch source metadata → **persist**) — **no text
extraction**. Design rationale and the external consult's findings live in the plan; this doc
is the build.

**Confirmed against the linked project `vumvzcugxbnkfnujpjjc` (Supabase, Postgres 17.6):**
`pgmq` **1.5.1** is available (includes the `queue_visible_length` metric KEDA needs);
`pg_cron` 1.6.4 available (not required). The `papers` table already exists as the catalog.

## The shape

```
submit (arxiv/doi/title) ─▶ upsert papers ─▶ publish_seed(paper_id)
                                                   │  (reads queue_subscriptions — the "SNS topic")
                          ┌────────────────┬───────┴────────────────┐
                          ▼                ▼                         ▼
                   harvest_arxiv    harvest_openalex           harvest_s2        ← one PGMQ queue per source
                          │                │                         │
                    ScaledJob arxiv   ScaledJob openalex        ScaledJob s2     ← KEDA, per-queue visible_length
                     max=1             max=3                     max=2
                          ▼                ▼                         ▼
          worker --source X:  take_token(X) → fetch source X → upsert source_X → harvest_attempts → archive
```

**One queue per harvest type**, and the seeder is decoupled from the queue list: it calls
`publish_seed(paper_id)`, which fans out to every queue registered in `queue_subscriptions`.
Adding a source = insert one subscription row, no seeder change. Each queue gets its own KEDA
`ScaledJob` with its own `maxReplicaCount`, so you scale sources independently (arXiv is
rate-limited hard → 1; OpenAlex's polite pool tolerates more → 3; etc.).

## Decisions (recap)

| Thing | Choice |
|---|---|
| Queue topology | **one PGMQ queue per source** (`harvest_arxiv`, `harvest_openalex`, `harvest_s2`) |
| Fan-out ("SNS") | `queue_subscriptions` registry + `publish_seed()` fn — seeder never enumerates queues; fan-out is atomic, in-DB |
| Autoscaler | KEDA **`ScaledJob`** per queue (one k8s Job per message) — **not** `ScaledObject` |
| KEDA metric | `SELECT queue_visible_length FROM pgmq.metrics('<queue>')` (excludes in-flight) |
| Per-type scale | independent `maxReplicaCount` per queue (arxiv 1, openalex 3, s2 2 — tune) |
| Catalog | reuse existing `papers` (identity + `harvest_status` enum) |
| Per-source data | **typed table per source** (`source_arxiv`, `source_openalex`, `source_semantic_scholar`) |
| Rate coordination | shared `rate_budget` token bucket (**required** once any `maxReplicaCount > 1`) |
| Worker DB access | `psycopg` → Supabase **session pooler, port 5432** (6543 forbids prepared statements) |
| Semantics | at-least-once + idempotent |
| IaC | Ansible + Helmfile, mirroring `aws-account-infra` `cluster_apps` |

**Type keys** are used consistently everywhere (`arxiv`, `openalex`, `s2`): queue =
`harvest_<type>`, `rate_budget.source = <type>`, `harvest_attempts.source = <type>`,
`queue_subscriptions.source = <type>`. The table `source_semantic_scholar` keeps its
descriptive name; the type key for it is `s2`.

---

## Repo 1 — Supabase migrations (`excalidraw-tf-rag/v3/supabase/migrations/`)

Convention (match `20260708141051_create_papers_table.sql`): `YYYYMMDDHHMMSS_<desc>.sql`, plain
non-idempotent SQL, RLS enabled with **no policies** (service-role only). Create these files
(bump the timestamps to now, keep them in this order).

### `…_enable_pgmq_and_queues.sql` — extension, queues, fan-out
```sql
create extension if not exists pgmq;

-- One queue per source. Each creates pgmq.q_<name> (live) + pgmq.a_<name> (archive).
select pgmq.create('harvest_arxiv');
select pgmq.create('harvest_openalex');
select pgmq.create('harvest_s2');

-- SNS-style subscription registry: adding a source = INSERT a row, no code change.
create table public.queue_subscriptions (
    queue_name text primary key,
    source     text not null,          -- type key: 'arxiv' | 'openalex' | 's2'
    active     boolean not null default true,
    filter     jsonb                   -- optional SNS-style filter policy (unused for now)
);
insert into public.queue_subscriptions (queue_name, source) values
    ('harvest_arxiv',    'arxiv'),
    ('harvest_openalex', 'openalex'),
    ('harvest_s2',       's2');

alter table public.queue_subscriptions enable row level security;

-- The "topic": fan one paper out to every active queue, atomically, in one transaction.
-- The seeder calls only this; it never names a queue.
create function public.publish_seed(p_paper_id uuid) returns void
language plpgsql as $$
declare q record;
begin
    for q in select queue_name from public.queue_subscriptions where active loop
        perform pgmq.send(q.queue_name, jsonb_build_object('paper_id', p_paper_id::text));
    end loop;
end $$;
```

### `…_create_source_arxiv.sql`
Mirrors `ArxivPaper` in `v3/harvest/src/harvest/sources/arxiv.py`.
```sql
create table public.source_arxiv (
    paper_id          uuid primary key references public.papers (paper_id) on delete cascade,
    entry_id          text not null,
    title             text not null,
    authors           text[] not null default '{}',
    summary           text,
    primary_category  text,
    categories        text[] not null default '{}',
    pdf_url           text,
    links             text[] not null default '{}',
    published         timestamptz,
    updated           timestamptz,
    comment           text,
    journal_ref       text,
    doi               text,
    fetched_at        timestamptz not null default now()
);

alter table public.source_arxiv enable row level security;
-- no policies: service-role/backend access only
```

### `…_create_source_openalex.sql`
Typed for the fields we use now; `raw jsonb` keeps everything else until promoted.
```sql
create table public.source_openalex (
    paper_id          uuid primary key references public.papers (paper_id) on delete cascade,
    openalex_id       text not null,
    doi               text,
    title             text,
    publication_date  date,
    type              text,
    best_oa_pdf_url   text,
    cited_by_count    integer,
    authorships       jsonb,
    raw               jsonb,
    fetched_at        timestamptz not null default now()
);

alter table public.source_openalex enable row level security;
```

### `…_create_source_semantic_scholar.sql`
```sql
create table public.source_semantic_scholar (
    paper_id             uuid primary key references public.papers (paper_id) on delete cascade,
    s2_paper_id          text,
    corpus_id            bigint,
    title                text,
    external_ids         jsonb,
    open_access_pdf_url  text,
    reference_count      integer,
    citation_count       integer,
    fields_of_study      text[],
    raw                  jsonb,
    fetched_at           timestamptz not null default now()
);

alter table public.source_semantic_scholar enable row level security;
```

### `…_create_harvest_attempts.sql`
Run history, separate from the catalog — every attempt per (paper, source), including failures.
This is your per-source state; the coarse `papers.status` enum can't express independent
per-source outcomes.
```sql
create table public.harvest_attempts (
    attempt_id   uuid primary key default gen_random_uuid(),
    paper_id     uuid references public.papers (paper_id) on delete cascade,
    source       text not null,                 -- type key: 'arxiv' | 'openalex' | 's2'
    queue_msg_id bigint,                         -- the pgmq msg id this attempt drained
    status       text not null,                 -- 'ok' | 'not_found' | 'error'
    error        text,
    started_at   timestamptz not null default now(),
    finished_at  timestamptz
);
create index harvest_attempts_paper_idx on public.harvest_attempts (paper_id, source);

alter table public.harvest_attempts enable row level security;
```

### `…_create_rate_budget.sql` — shared token bucket (needed once any `maxReplicaCount > 1`)
Pod count ≠ request rate. When N pods of one source share that source's rate limit, they all
draw tokens from one bucket. A worker calls `take_token('<source>')` in a backoff loop until it
gets `true`, then makes the API call.
```sql
create table public.rate_budget (
    source          text primary key,           -- type key, matches queue_subscriptions.source
    tokens          numeric not null,           -- current available tokens
    burst           numeric not null,           -- bucket capacity (max tokens)
    refill_per_sec  numeric not null,           -- tokens added per second
    last_refill     timestamptz not null default now()
);
insert into public.rate_budget (source, tokens, burst, refill_per_sec) values
    ('arxiv',    1,  1,  0.33),                 -- ~1 req / 3s
    ('openalex', 10, 10, 10),                   -- polite pool ~10 req/s
    ('s2',       1,  1,  1);                     -- ~1 req/s unauthenticated

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

alter table public.rate_budget enable row level security;
```

### `…_harvester_roles.sql` — least-privilege DB roles
Two roles: the worker (read/write + full pgmq + fan-out/rate helpers) and the KEDA scaler
(**read-only**, metrics only).
```sql
-- Worker: catalog + source tables + pgmq + fan-out + rate bucket.
create role harvester_worker login password 'REPLACE_ME_WORKER';
grant usage on schema public, pgmq to harvester_worker;
grant select, insert, update on
    public.papers, public.source_arxiv, public.source_openalex,
    public.source_semantic_scholar, public.harvest_attempts
    to harvester_worker;
grant select on public.queue_subscriptions to harvester_worker;
grant select, update on public.rate_budget to harvester_worker;
grant execute on function public.publish_seed(uuid)  to harvester_worker;
grant execute on function public.take_token(text)    to harvester_worker;
grant execute on all functions in schema pgmq to harvester_worker;
grant select, insert, update, delete on all tables in schema pgmq to harvester_worker;

-- KEDA scaler: read-only, only needs pgmq.metrics().
create role harvester_keda_ro login password 'REPLACE_ME_KEDA';
grant usage on schema pgmq to harvester_keda_ro;
grant execute on function pgmq.metrics(text) to harvester_keda_ro;
grant select on all tables in schema pgmq to harvester_keda_ro;
```
> Set real passwords out-of-band and store them only in Ansible Vault (Repo 3). Rotate the
> secrets currently sitting in `v3/harvest/.env`.

### `…_create_seeds.sql` — *optional* submission inbox
Only if you want submission decoupled from resolution. Canonical IDs (arxiv/doi/openalex)
resolve trivially; a **title** needs an API call to resolve to a `paper_id`. If the seeder
resolves synchronously, skip this table.
```sql
create table public.seeds (
    seed_id      uuid primary key default gen_random_uuid(),
    id_type      text not null,                 -- 'arxiv' | 'doi' | 'openalex' | 'title'
    id_value     text not null,                 -- normalized value
    raw_input    text not null,                 -- exactly what was submitted
    status       text not null default 'new',   -- 'new' | 'resolved' | 'unresolvable'
    paper_id     uuid references public.papers (paper_id),
    note         text,
    submitted_at timestamptz not null default now(),
    resolved_at  timestamptz,
    unique (id_type, id_value)
);

alter table public.seeds enable row level security;
```

**Apply** (either):
- `cd v3/supabase && supabase db push` (applies the new files to the linked project), or
- Supabase MCP `apply_migration` per file.

**Verify:**
```sql
select queue_visible_length from pgmq.metrics('harvest_arxiv');   -- returns a row
select public.publish_seed(gen_random_uuid());                    -- fans out to 3 queues
select queue_name, source from public.queue_subscriptions;        -- 3 rows
```
`\d public.source_arxiv` shows the typed columns.

---

## Repo 2 — Harvest worker + seeder (`excalidraw-tf-rag/v3/harvest/`)

### `pyproject.toml`
Add the driver and two entrypoints:
```toml
[project]
dependencies = [
    # …existing…
    "psycopg[binary]>=3.2",
]

[project.scripts]
harvest      = "harvest:main"            # keep the existing one-shot CLI for now
harvest-seed = "harvest.seeder:main"
harvest-work = "harvest.worker:main"
```
Then `cd v3/harvest && uv lock` (CI's `lock-check` fails on drift).

### `src/harvest/settings.py`
Add the DB DSN (session pooler) and a `source` selector (the per-type worker reads it — the
`ScaledJob` passes `--source <type>`).
```python
from urllib.parse import quote


class HarvestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", cli_parse_args=True, extra="ignore"
    )
    # seeds (on-demand submit)
    arxiv_id: str | None = Field(default=None, description="Seed arXiv identifier")
    doi: str | None = None
    # worker selector: which source this Job harvests ('arxiv' | 'openalex' | 's2')
    source: str | None = None
    # storage / API
    supabase_url: str | None = None
    supabase_key: str | None = None
    openalex_api_key: str | None = None
    # DB (session pooler, 5432)
    harvest_db_dsn: str | None = None  # full DSN incl. password (k8s Secret)
    supabase_pooler_url: str | None = None  # postgresql://user@host:5432/postgres
    supabase_staging_pw: str | None = None

    @property
    def db_dsn(self) -> str:
        if self.harvest_db_dsn:
            return self.harvest_db_dsn
        if self.supabase_pooler_url and self.supabase_staging_pw:
            scheme, rest = self.supabase_pooler_url.split("://", 1)
            userinfo, hostpart = rest.split("@", 1)
            user = userinfo.split(":", 1)[0]
            return f"{scheme}://{user}:{quote(self.supabase_staging_pw)}@{hostpart}"
        raise RuntimeError(
            "No DB DSN: set HARVEST_DB_DSN or SUPABASE_POOLER_URL+SUPABASE_STAGING_PW"
        )
```

### `src/harvest/db.py` (new) — psycopg + PGMQ (per-queue)
```python
"""Thin psycopg helpers: per-queue PGMQ ops, fan-out, rate bucket, typed upserts.

Session pooler (5432). One queue per source; the type key selects the queue.
"""

import logging
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

logger = logging.getLogger("harvest.db")

# type key -> queue name. Keep in sync with queue_subscriptions.
QUEUE_FOR = {
    "arxiv": "harvest_arxiv",
    "openalex": "harvest_openalex",
    "s2": "harvest_s2",
}


def connect(dsn: str) -> psycopg.Connection:
    # autocommit=False: explicit transactions for publish + ack.
    return psycopg.connect(dsn, row_factory=dict_row, autocommit=False)


# ---- Seeder side: upsert identity + fan out via publish_seed ---------------
def submit(
    conn: psycopg.Connection,
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    force: bool = False,
) -> str | None:
    """Upsert identity into papers and fan out to every subscribed queue, atomically.

    Returns the paper_id if it was published, else None (already known and not forced).
    The seeder does NOT name any queue — publish_seed() reads queue_subscriptions.
    """
    if not arxiv_id and not doi:
        raise ValueError("submit requires arxiv_id or doi")
    col, val = ("arxiv_id", arxiv_id) if arxiv_id else ("doi", doi)
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                f"insert into papers ({col}) values (%s) "
                f"on conflict ({col}) do nothing returning paper_id",
                (val,),
            )
            row = cur.fetchone()
            if row is None:
                if not force:
                    return None
                cur.execute(f"select paper_id from papers where {col} = %s", (val,))
                row = cur.fetchone()
            paper_id = str(row["paper_id"])
            cur.execute("select public.publish_seed(%s)", (paper_id,))
    return paper_id


# ---- Worker side: per-queue read / heartbeat / ack -------------------------
def read_one(conn: psycopg.Connection, queue: str, vt: int = 120) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute("select * from pgmq.read(%s, %s, 1)", (queue, vt))
        return cur.fetchone()  # msg_id, read_ct, enqueued_at, vt, message


def heartbeat(conn: psycopg.Connection, queue: str, msg_id: int, vt: int = 120) -> None:
    with conn.cursor() as cur:
        cur.execute("select pgmq.set_vt(%s, %s, %s)", (queue, msg_id, vt))


def ack(conn: psycopg.Connection, queue: str, msg_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("select pgmq.archive(%s, %s)", (queue, msg_id))


def visible_len(conn: psycopg.Connection, queue: str) -> int:  # parity w/ KEDA metric
    with conn.cursor() as cur:
        cur.execute("select queue_visible_length from pgmq.metrics(%s)", (queue,))
        return cur.fetchone()["queue_visible_length"]


# ---- Identity lookup (payload carries only paper_id) -----------------------
def get_identity(conn: psycopg.Connection, paper_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "select arxiv_id, doi, openalex_id from papers where paper_id = %s",
            (paper_id,),
        )
        return cur.fetchone()


# ---- Shared rate bucket ----------------------------------------------------
def acquire_token(conn: psycopg.Connection, source: str, *, timeout: float = 60.0) -> bool:
    """Block (with backoff) until a token is free for `source`, or give up after timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with conn.cursor() as cur:
            cur.execute("select public.take_token(%s) as ok", (source,))
            if cur.fetchone()["ok"]:
                conn.commit()
                return True
        conn.commit()
        time.sleep(0.5)
    return False


# ---- Typed per-source upserts ----------------------------------------------
def upsert_source_arxiv(conn: psycopg.Connection, paper_id: str, p) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into source_arxiv (paper_id, entry_id, title, authors, summary,
                primary_category, categories, pdf_url, links, published, updated,
                comment, journal_ref, doi, fetched_at)
            values (%(pid)s, %(entry_id)s, %(title)s, %(authors)s, %(summary)s,
                %(primary_category)s, %(categories)s, %(pdf_url)s, %(links)s,
                %(published)s, %(updated)s, %(comment)s, %(journal_ref)s, %(doi)s, now())
            on conflict (paper_id) do update set
                entry_id=excluded.entry_id, title=excluded.title, authors=excluded.authors,
                summary=excluded.summary, primary_category=excluded.primary_category,
                categories=excluded.categories, pdf_url=excluded.pdf_url, links=excluded.links,
                published=excluded.published, updated=excluded.updated, comment=excluded.comment,
                journal_ref=excluded.journal_ref, doi=excluded.doi, fetched_at=now()
            """,
            {
                "pid": paper_id,
                "entry_id": p.entry_id,
                "title": p.title,
                "authors": p.authors,
                "summary": p.summary,
                "primary_category": p.primary_category,
                "categories": p.categories,
                "pdf_url": p.pdf_url,
                "links": p.links,
                "published": p.published,
                "updated": p.updated,
                "comment": p.comment,
                "journal_ref": p.journal_ref,
                "doi": p.doi,
            },
        )


def upsert_source_semantic_scholar(
    conn,
    paper_id: str,
    *,
    s2_paper_id=None,
    title=None,
    external_ids=None,
    open_access_pdf_url=None,
    reference_count=None,
    citation_count=None,
    fields_of_study=None,
    raw=None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into source_semantic_scholar (paper_id, s2_paper_id, title, external_ids,
                open_access_pdf_url, reference_count, citation_count, fields_of_study, raw, fetched_at)
            values (%(pid)s, %(s2)s, %(title)s, %(ext)s, %(pdf)s, %(rc)s, %(cc)s, %(fos)s, %(raw)s, now())
            on conflict (paper_id) do update set
                s2_paper_id=excluded.s2_paper_id, title=excluded.title,
                external_ids=excluded.external_ids, open_access_pdf_url=excluded.open_access_pdf_url,
                reference_count=excluded.reference_count, citation_count=excluded.citation_count,
                fields_of_study=excluded.fields_of_study, raw=excluded.raw, fetched_at=now()
            """,
            {
                "pid": paper_id,
                "s2": s2_paper_id,
                "title": title,
                "ext": Json(external_ids) if external_ids is not None else None,
                "pdf": open_access_pdf_url,
                "rc": reference_count,
                "cc": citation_count,
                "fos": fields_of_study,
                "raw": Json(raw) if raw is not None else None,
            },
        )


# (write upsert_source_openalex the same way once OpenAlexSource lands)


def record_attempt(conn, paper_id, source, status, *, queue_msg_id=None, error=None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "insert into harvest_attempts (paper_id, source, queue_msg_id, status, error, finished_at) "
            "values (%s, %s, %s, %s, %s, now())",
            (paper_id, source, queue_msg_id, status, error),
        )


def set_paper_status(conn, paper_id, status) -> None:  # 'succeeded' | 'exhausted'
    with conn.cursor() as cur:
        cur.execute(
            "update papers set status=%s, harvested_at=now() where paper_id=%s",
            (status, paper_id),
        )
```

### `src/harvest/seeder.py` (new)
One write + one fan-out. Never names a queue.
```python
import logging, sys
from .settings import HarvestSettings
from . import db

logging.basicConfig(
    level="INFO", stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("harvest.seeder")


def main() -> None:
    s = HarvestSettings()
    with db.connect(s.db_dsn) as conn:
        # On-demand submit. (Discovery-from-OpenAlex "since watermark" goes here later.)
        paper_id = db.submit(conn, arxiv_id=s.arxiv_id, doi=s.doi)
    if paper_id:
        log.info("published paper_id=%s to all subscribed queues", paper_id)
    else:
        log.info("already known, skipped")
```

### `src/harvest/worker.py` (new) — one source per Job
The `ScaledJob` for source X runs `harvest-work --source X`. The worker reads only queue
`harvest_X`, harvests only source X, writes only `source_X`. Drain-then-exit; one Job per
message. Ordering: acquire token → fetch → persist source row → record attempt → catalog
status → `ack`, so a crash before ack simply redelivers and converges.
```python
import logging, sys
from .settings import HarvestSettings
from .sources import ArxivSource, SemanticScholarSource
from . import db

logging.basicConfig(
    level="INFO", stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("harvest.worker")


def harvest_arxiv(conn, paper_id, ident, msg_id) -> bool:
    if not db.acquire_token(conn, "arxiv"):
        raise TimeoutError("no arxiv rate token")
    ap = ArxivSource().get_arxiv_paper(ident["arxiv_id"])  # see arxiv.py change below
    if ap is None:
        db.record_attempt(conn, paper_id, "arxiv", "not_found", queue_msg_id=msg_id)
        return False
    db.upsert_source_arxiv(conn, paper_id, ap)
    db.record_attempt(conn, paper_id, "arxiv", "ok", queue_msg_id=msg_id)
    return True


def harvest_s2(conn, paper_id, ident, msg_id) -> bool:
    if not db.acquire_token(conn, "s2"):
        raise TimeoutError("no s2 rate token")
    src = SemanticScholarSource()
    paper = src.get_paper(ident["arxiv_id"])  # S2 keyed off ARXIV:<id>
    if paper is None:
        db.record_attempt(conn, paper_id, "s2", "not_found", queue_msg_id=msg_id)
        return False
    db.upsert_source_semantic_scholar(
        conn,
        paper_id,
        title=paper.title,
        open_access_pdf_url=paper.pdf_url,
        external_ids={"ArXiv": paper.arxiv_id, "DOI": paper.doi},
    )
    db.record_attempt(conn, paper_id, "s2", "ok", queue_msg_id=msg_id)
    return True


HARVESTERS = {"arxiv": harvest_arxiv, "s2": harvest_s2}  # add "openalex" when it lands


def main() -> None:
    s = HarvestSettings()
    source = s.source
    if source not in db.QUEUE_FOR:
        raise SystemExit(f"--source must be one of {sorted(db.QUEUE_FOR)}, got {source!r}")
    queue = db.QUEUE_FOR[source]
    harvest = HARVESTERS.get(source)
    if harvest is None:
        raise SystemExit(f"no harvester wired for source={source!r}")

    with db.connect(s.db_dsn) as conn:
        while True:
            rec = db.read_one(conn, queue, vt=120)
            if rec is None:
                log.info("queue %s empty, exiting", queue)
                return
            msg, msg_id = rec["message"], rec["msg_id"]
            paper_id = msg["paper_id"]
            try:
                ident = db.get_identity(conn, paper_id)
                ok = harvest(conn, paper_id, ident, msg_id)
                db.set_paper_status(conn, paper_id, "succeeded" if ok else "exhausted")
                conn.commit()
                db.ack(conn, queue, msg_id)
                conn.commit()
                log.info("done source=%s msg_id=%s paper_id=%s ok=%s", source, msg_id, paper_id, ok)
            except Exception:
                conn.rollback()  # leave message unacked -> redelivered after vt
                log.exception("failed source=%s msg_id=%s; will redeliver", source, msg_id)
```
> `set_paper_status` writes the *catalog* status. With independent per-source queues, a paper is
> "succeeded" as soon as any source persists; per-source truth lives in `source_*` rows +
> `harvest_attempts`. Refine the catalog rollup later (e.g. only mark `exhausted` when all
> sources report not_found) — the attempts table has what you need.

### `src/harvest/sources/arxiv.py` — fixes + typed accessor
1. Return the rich record: add
   ```python
   def get_arxiv_paper(self, identifier: str) -> "ArxivPaper | None":
       return self._fetch(identifier)  # _fetch already returns ArxivPaper
   ```
   Keep `get_paper()` returning the canonical `Paper` for callers that want it.
2. In `_fetch`, **remove** the stray debug lines (`logger.info("search: %s"…)`,
   `logger.info("results: %s"…)`, the `arxiv_paper` pretty-print) and the **unreachable code**
   after the `try/except` (the trailing `logger.info(...)`/`return paper`). `_fetch` should
   build and return `ArxivPaper.from_result(next(results))`, or `None` on
   `StopIteration`/HTTP errors. Drop the now-unused `import pprint`.

### `src/harvest/sources/semantic_scholar.py` — bug fix
Line 29 calls `pprint.pformat(...)` but `pprint` is **not imported** → `NameError` on the happy
path. Delete that debug line (preferred) or `import pprint` at top.

### `Dockerfile`
The image already `ENTRYPOINT ["harvest"]`. For k8s, override the command per workload instead
of baking a new entrypoint: each `ScaledJob` sets `command: ["harvest-work", "--source", "<type>"]`
and the `CronJob` sets `command: ["harvest-seed"]` (both are console scripts on `PATH`). No
Dockerfile change needed beyond rebuilding after the dependency add.

### Local end-to-end (no k8s)
```bash
cd v3/harvest && uv sync
uv run harvest-seed --arxiv_id 1706.03762     # -> "published paper_id=… to all subscribed queues"
uv run harvest-work --source arxiv            # drains harvest_arxiv, persists source_arxiv, archives
uv run harvest-work --source s2               # drains harvest_s2, persists source_semantic_scholar
```
Verify: a `papers` row (`status='succeeded'`), a `source_arxiv` + `source_semantic_scholar` row,
`harvest_attempts` rows, and `queue_visible_length` back to 0 for each queue. **Crash test:**
`kill` a `harvest-work` between persist and ack; re-run and confirm it converges (no dup, ends
`done`).

---

## Repo 3 — Ansible/Helm (`aws-account-infra/ansible/`)

> [!NOTE]
> **Reconciled build (2026-07-11).** The sketch below predates the shipped worker; the
> actual deployment (the `cluster_harvester` role in `aws-account-infra`) differs on a few
> points — trust these where they disagree:
> - **Worker transport is Supabase REST** (`pgmq_public`), so the pod env is
>   `SUPABASE_URL` + `SUPABASE_SERVICE_KEY`, **not** a worker DB DSN. There is no
>   `harvester_worker` DB role — only `harvester_keda_ro` (for the KEDA scaler).
> - **Image is the versioned prod tag**
>   `992382747916.dkr.ecr.us-east-1.amazonaws.com/excalidraw-tf-rag/v3/harvest/prod:<ver>`
>   (published on merge to master). The container `ENTRYPOINT` is `["harvest"]`, so the
>   ScaledJob overrides `command: ["harvest-work","--source","<src>"]`.
> - **ECR pull** uses a node **kubelet credential provider** fed by a scoped read-only IAM
>   user (provisioned in Terraform) — **no `imagePullSecret`**, no refresh CronJob.
> - Schema is the current `seeds`/bigint model; `publish_seed(bigint)`.

Mirror `roles/cluster_apps` exactly: Helmfile via `command`, raw manifests via
Jinja→`kubectl apply`, vault→`kubectl create secret --dry-run|apply`, versions pinned in
`defaults/main.yml`.

### 3a. Install KEDA as a `cluster_addons` release (platform layer)
In `roles/cluster_addons/templates/helmfile.yaml.j2` add the repo + release:
```yaml
repositories:
  # …existing…
  - name: kedacore
    url: https://kedacore.github.io/charts

releases:
  # …existing…
  - name: keda
    namespace: keda
    createNamespace: true
    chart: kedacore/keda
    version: "{{ cluster_addons_keda_version }}"
```
In `roles/cluster_addons/defaults/main.yml` add (pin to the current chart — **verify latest**):
```yaml
cluster_addons_keda_version: "2.15.1"   # chart for KEDA 2.20; verify before applying
```
Add a readiness wait in `roles/cluster_addons/tasks/main.yml` after the helmfile apply:
```yaml
- name: Wait for KEDA operator
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_addons_kubeconfig }} wait deployment/keda-operator
    --namespace keda --for=condition=Available --timeout=5m
  changed_when: false
```

### 3b. New role `roles/cluster_harvester/`

**`defaults/main.yml`** — the per-source scaling knobs live here as a list; the template loops
over it to emit one `ScaledJob` per source.
```yaml
---
cluster_harvester_kubeconfig: "{{ lookup('env', 'HOME') }}/.kube/config"
cluster_harvester_work_dir: /tmp/homelab-kubernetes-harvester
cluster_harvester_namespace: harvester
cluster_harvester_secret: harvester-secrets

# Pinned. Keep pinned per repo policy.
cluster_harvester_image: "{{ kubernetes_harvester_image }}"   # e.g. <ecr>/harvest:<tag>

# One ScaledJob per source. queue must match pgmq.create() + queue_subscriptions.
# max_replicas is the per-source parallelism cap (respect each source's rate limit).
cluster_harvester_sources:
  - { name: arxiv,    queue: harvest_arxiv,    max_replicas: 1 }
  - { name: openalex, queue: harvest_openalex, max_replicas: 3 }
  - { name: s2,       queue: harvest_s2,       max_replicas: 2 }

cluster_harvester_target_value: "1"   # 1 Job per pending message, capped at max_replicas
```

**`tasks/main.yml`** (mirrors `cluster_apps/tasks/main.yml`)
```yaml
---
- name: Assert local cluster tools are installed
  ansible.builtin.command: "{{ item }} version"
  loop: [kubectl, helm, helmfile]
  changed_when: false

- name: Create harvester working directory
  ansible.builtin.file:
    path: "{{ cluster_harvester_work_dir }}"
    state: directory
    mode: "0700"

- name: Ensure the harvester namespace exists
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} create namespace
    {{ cluster_harvester_namespace }} --dry-run=client --output yaml
  register: cluster_harvester_ns
  changed_when: false
- name: Apply the harvester namespace
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} apply --filename -
  args: { stdin: "{{ cluster_harvester_ns.stdout }}" }
  register: cluster_harvester_ns_apply
  changed_when: "'unchanged' not in cluster_harvester_ns_apply.stdout"

- name: Render the harvester secret from Vault
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} create secret generic
    {{ cluster_harvester_secret }} --namespace {{ cluster_harvester_namespace }}
    --from-literal=HARVEST_DB_DSN={{ vault_harvester_db_url }}
    --from-literal=KEDA_DB_DSN={{ vault_harvester_keda_dsn }}
    --from-literal=OPENALEX_API_KEY={{ vault_harvester_openalex_key | default('') }}
    --dry-run=client --output yaml
  register: cluster_harvester_secret_manifest
  changed_when: false
  no_log: true
  when: vault_harvester_db_url is defined
- name: Apply the harvester secret
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} apply --filename -
  args: { stdin: "{{ cluster_harvester_secret_manifest.stdout }}" }
  no_log: true
  changed_when: true
  when: vault_harvester_db_url is defined
- name: Assert the harvester secret exists
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} get secret
    {{ cluster_harvester_secret }} --namespace {{ cluster_harvester_namespace }}
  changed_when: false

- name: Render harvester resources
  ansible.builtin.template:
    src: harvester-resources.yml.j2
    dest: "{{ cluster_harvester_work_dir }}/harvester-resources.yml"
    mode: "0600"
- name: Apply harvester resources
  ansible.builtin.command: >-
    kubectl --kubeconfig {{ cluster_harvester_kubeconfig }} apply
    --filename {{ cluster_harvester_work_dir }}/harvester-resources.yml
  register: cluster_harvester_apply
  changed_when: "'unchanged' not in cluster_harvester_apply.stdout"
```

**`templates/harvester-resources.yml.j2`** — shared TriggerAuth + **one ScaledJob per source**
(loop) + discovery CronJob.
```yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: keda-pg-auth
  namespace: {{ cluster_harvester_namespace }}
type: Opaque
stringData:
  # Read-only DSN the KEDA scaler uses (session pooler, 5432).
  connection: "{{ vault_harvester_keda_dsn }}"
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: harvester-pg
  namespace: {{ cluster_harvester_namespace }}
spec:
  secretTargetRef:
    - parameter: connection
      name: keda-pg-auth
      key: connection
{% for src in cluster_harvester_sources %}
---
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: harvester-{{ src.name }}
  namespace: {{ cluster_harvester_namespace }}
spec:
  jobTargetRef:
    backoffLimit: 3
    template:
      spec:
        restartPolicy: Never
        containers:
          - name: harvest-work
            image: {{ cluster_harvester_image }}
            command: ["harvest-work", "--source", "{{ src.name }}"]
            envFrom:
              - secretRef:
                  name: {{ cluster_harvester_secret }}
            resources:
              requests: { cpu: "100m", memory: "256Mi" }
              limits:   { cpu: "500m", memory: "512Mi" }
  pollingInterval: 15
  maxReplicaCount: {{ src.max_replicas }}
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 10
  rollout:
    strategy: gradual
  scalingStrategy:
    strategy: accurate
  triggers:
    - type: postgresql
      metadata:
        query: "SELECT queue_visible_length FROM pgmq.metrics('{{ src.queue }}')"
        targetQueryValue: "{{ cluster_harvester_target_value }}"
      authenticationRef:
        name: harvester-pg
{% endfor %}
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: harvester-discovery
  namespace: {{ cluster_harvester_namespace }}
spec:
  schedule: "0 * * * *"          # hourly discovery; tune later
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: harvest-seed
              image: {{ cluster_harvester_image }}
              command: ["harvest-seed"]     # add discovery args when implemented
              envFrom:
                - secretRef:
                    name: {{ cluster_harvester_secret }}
```
> Each `postgresql` scaler `query` returns a single numeric cell (`queue_visible_length` for that
> source's queue). The DSN must point at the **session pooler (5432)**; pgx (which the scaler
> uses) prepares statements, and Supabase's 6543 transaction pooler rejects those.
>
> **Rate-limit reminder:** `maxReplicaCount` caps *pods*, not *request rate*. Any source with
> `max_replicas > 1` (openalex, s2 here) relies on the `rate_budget` token bucket so N pods don't
> collectively exceed the source's limit. Set each `max_replicas` ≈
> `floor(source_rate_limit / per_pod_safe_rate)` so caps and bucket agree.

### 3c. Playbook `playbooks/harvester.yml`
```yaml
---
- name: Deploy the harvester workload
  hosts: admin_workstations
  become: false
  gather_facts: true
  roles:
    - cluster_harvester
```

### 3d. Inventory + Vault
- `inventories/example/group_vars/all.yml` — documented placeholders:
  ```yaml
  kubernetes_harvester_image: "192.0.2.1/harvest:REPLACE"
  ```
- `inventories/local/group_vars/all.yml` (gitignored) — real image ref.
- Vault file (gitignored, `inventories/local/group_vars/`) — secrets:
  ```yaml
  vault_harvester_db_url: "postgresql://harvester_worker:PW@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
  vault_harvester_keda_dsn: "postgresql://harvester_keda_ro:PW@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
  vault_harvester_openalex_key: "…"   # optional
  ```

### 3e. Image publishing
The harvester image builds/pushes to ECR **on-demand via the workload repo's CI** (per
`live/aws/staging/workloads/rag`) — no new `deploy-units.json` unit. Set
`kubernetes_harvester_image` to that repo:tag.

### 3f. Governance note
Add `aws-account-infra/docs/plans/keda.md` recording the decision to add the KEDA addon (the
repo's `CLAUDE.md` gates new k8s installs behind a reviewed plan). Short is fine: what, why
(harvester per-source autoscaling), how (pinned `cluster_addons` release), rollback
(`helm uninstall keda`).

### 3g. Deploy
```bash
cd aws-account-infra
ANSIBLE_CONFIG=ansible/ansible.cfg ansible-playbook ansible/playbooks/addons.yml     # installs KEDA
ANSIBLE_CONFIG=ansible/ansible.cfg ansible-playbook ansible/playbooks/harvester.yml  # role
```

---

## End-to-end verification (cluster)
1. `kubectl get crd | grep keda.sh` → KEDA CRDs present.
2. `kubectl -n harvester get scaledjob,cronjob,secret` → three ScaledJobs
   (`harvester-arxiv/-openalex/-s2`), the CronJob, and the secret.
3. Seed a message (locally via `harvest-seed`, or wait for the CronJob) → one message lands in
   **each** subscribed queue.
4. `kubectl -n harvester get jobs -w` → per source, a Job spawns while its
   `queue_visible_length > 0`, runs to completion, and no new Jobs once that queue drains (scale
   to zero). Confirm the per-source caps hold (openalex can reach 3 concurrent Jobs, arxiv 1).
5. Confirm rows in `papers` / `source_arxiv` / `source_semantic_scholar` / `harvest_attempts`;
   all queues back to 0.
6. Kill a running Job mid-flight → the message reappears after its visibility timeout and a new
   Job converges it (at-least-once, idempotent).

## Adding a fourth source later (the payoff of decoupling)
1. `select pgmq.create('harvest_<new>');` + `insert into queue_subscriptions …` (migration).
2. `insert into rate_budget …`; add an `upsert_source_<new>` + a harvester fn in `worker.py`;
   create `source_<new>` table.
3. Append one entry to `cluster_harvester_sources` in `defaults/main.yml`.
The **seeder never changes** — `publish_seed()` picks up the new subscription automatically.

## Deferred (not in this build)
Text extraction (GROBID/pdfplumber), OpenAlex discovery-since-watermark + `OpenAlexSource`,
S3 artifact backup, fetch/extract stage split, title→paper resolution via the `seeds` inbox,
lifting the fan-out topic into NATS JetStream (only if you outgrow Postgres-as-broker — the
per-source KEDA `ScaledJob`s stay identical), Argo.
