# Harvest dispatch: capability-gated fan-out

How a seed reaches the sources that can actually harvest it, exactly once, and how to
make it happen again on purpose.

Companion to [`harvest-implementation.md`](harvest-implementation.md) (the build) and
[`harvest-worker-scaling.md`](harvest-worker-scaling.md) (worker semantics). Implemented
by migrations `20260716223236_harvest_dispatch` (tables) and
`20260716223300_capability_gated_fanout` (behaviour).

## The problem this fixes

`publish_seed` used to enqueue to **every** active queue unconditionally. That is wrong
because **the sources do not accept the same identifiers**:

| source | accepts | never accepts |
|---|---|---|
| arxiv | arXiv id | anything else |
| s2 | `ARXIV:<id>` or `DOI:<doi>` | — |
| openalex | DOI or OpenAlex id | **arXiv id** |

OpenAlex has no arXiv filter, and arXiv's own DOI (`10.48550/arxiv.<id>`) does not
resolve there — both lookup forms return 404 (verified 2026-07-16). So an arXiv-only
seed landed in `harvest_openalex`, the worker could not possibly succeed, and it wrote
`status='not_found'`.

**That was a lie, and it is the whole point of this design.** The paper was not missing
from OpenAlex; we never held an identity to ask with. Those two states need different
responses — one is final, the other resolves itself the moment a DOI arrives — and the
old schema could not tell them apart.

Now: `harvest_dispatch` records that we **asked**; `source_*.status` records what we
**learned**. "Never dispatched" is representable, and distinct from `not_found`.

## The two gates

`publish_seed(paper_id)` asks two questions per subscription:

1. **Can this source run?** — `seed_satisfies(seed, subscription.filter)`. The filter
   (`{"any_of": ["doi", "openalex_id"]}`) declares the identity the source needs. Stored
   in `queue_subscriptions.filter`, which existed as *"optional SNS-style filter policy
   (unused for now)"* from the start.
2. **Has it already?** — `harvest_claim_dispatch(paper_id, source)`. A row in
   `harvest_dispatch` keyed `(paper_id, source)`. **Only a caller that wins the claim
   sends a message.**

Gate 2 is why the trigger below is safe. Re-publishing becomes a cheap no-op instead of
a stampede.

### Dedupe happens at enqueue, not consume

By the time a message exists, KEDA has spawned a Job and the API call is paid for. So
the dedupe is a primary-key conflict inside the same transaction as the identity write:

```sql
insert into harvest_dispatch (paper_id, source, generation)
values (...)
on conflict (paper_id, source) do update
    set generation = excluded.generation, ...
    where harvest_dispatch.generation < excluded.generation
returning true          -- send ONLY if this returned a row
```

Race safety is the `ON CONFLICT` row lock, not application logic. Two concurrent calls
serialise; the loser's `WHERE` is false, so no row is updated, `RETURNING` yields
nothing, and it returns false. Exactly one message is sent.

This is stronger than SQS FIFO dedup (a 5-minute window). It is exactly-once per
`(paper_id, source)` per generation, permanently.

> Do **not** rewrite the claim as SELECT-then-INSERT. That reintroduces the race it exists
> to close.

## The three flows

Verified end-to-end against staging on 2026-07-16.

### 1. New seed — arXiv id only

```sql
insert into seeds (corpus, arxiv_id) values ('ml', '1706.03762');   -- paper_id 8
```

`AFTER INSERT` → `publish_seed(8)`:

| source | filter satisfied? | claim | outcome |
|---|---|---|---|
| arxiv | ✅ | insert ok | **send** |
| s2 | ✅ | insert ok | **send** |
| openalex | ❌ (no doi/openalex_id) | *not called* | nothing |

```
harvest_dispatch → {arxiv: gen1, s2: gen1}      -- no openalex row at all
```

KEDA sees `harvest_openalex` depth 0 → **no Job, no container**. The absent row *is* the
honest answer.

### 2. Identity grows — S2 discovers a DOI

The s2 worker writes what it learned:

```sql
update seeds set doi = '10.18653/v1/N19-1423' where paper_id = 8;
```

`BEFORE` trigger applies `coalesce` (monotonic); `AFTER UPDATE ... WHEN` sees
`NULL IS DISTINCT FROM '10.18653…'` → fires → `publish_seed(8)`:

| source | satisfied? | claim | outcome |
|---|---|---|---|
| arxiv | ✅ | conflict, `gen1 < gen1` = false | **no send** |
| s2 | ✅ | conflict, false | **no send** |
| openalex | ✅ **now** | no row existed → insert ok | **send** |

```
harvest_dispatch → {arxiv: gen1, openalex: gen1, s2: gen1}
```

The cascade: `arxiv_id → s2 → doi → openalex`. arXiv and S2 do not re-run even though the
trigger fired — the claim absorbs it.

**The trigger makes the cascade a property of the data, not of worker discipline.** A
worker that forgets to re-publish cannot break it.

### 2b. Redelivery is free

If a worker crashes after `update seeds` but before archiving its message, pgmq
redelivers. The worker re-runs, upserts `source_s2` (idempotent), and writes the **same**
DOI. `coalesce` keeps it, so `OLD.doi IS DISTINCT FROM NEW.doi` is **false** and the
trigger does not fire. Verified: dispatch state unchanged.

Without the `WHEN` clause every redelivery would re-run the fan-out.

### 3. Re-run one source after a change

```sql
select public.harvest_republish('openalex', 'fixed authorships extractor');
```

| source | claim check | outcome |
|---|---|---|
| arxiv | `gen1 < gen1` false | **no send** |
| s2 | `gen1 < gen1` false | **no send** |
| openalex | `gen1 < gen2` **true** → update | **send** |

```
harvest_dispatch → {arxiv: gen1, openalex: gen2, s2: gen1}
```

Only OpenAlex re-runs, and only for seeds that can run it. **arXiv's 1-request/3-second
budget is never touched** — which is why `generation` is per-source rather than global.

**The sweep is not optional.** The trigger fires on `seeds` changes; bumping a generation
does not touch `seeds`, so nothing would re-dispatch. `harvest_republish()` wraps the
bump and the sweep together so that cannot be forgotten.

Single paper instead of the corpus:

```sql
delete from harvest_dispatch where paper_id = 8 and source = 'openalex';
select public.publish_seed(8);
```

## Why it terminates

**Identity is monotonic** — fields go `NULL → value` and never change or clear, enforced
by a `BEFORE UPDATE` trigger (`new.doi := coalesce(old.doi, new.doi)`).

Therefore each seed fires the trigger **at most once per identity field** (4 fields → ≤4
cascades), and each source is claimed at most once per generation. The cascade is
**provably** bounded, not probably bounded.

It breaks the instant something overwrites an identity field — which is why the
`coalesce` trigger is load-bearing rather than defensive.

## Who may write identity

**Only sources whose identity claims we measured as trustworthy: arXiv, S2, Crossref.**

**OpenAlex is data-only. It must never write `seeds`.** Measured on a 5-paper sample:

- 2 of 5 title lookups returned an entirely different paper (0/4 and 0/10 author overlap)
- Its DOI for "Attention Is All You Need" is a mirror's (`10.65215/…`), not the paper's
- Its `cited_by_count` for that paper is 6,587 against S2's 184,545 — a 28× undercount

A bad DOI written to `seeds` would satisfy the Crossref and Unpaywall filters and send
them to fetch **the wrong paper's data**, persisted as canonical. One bad identity
poisons every downstream source. The `coalesce` trigger blocks the overwrite case; the
policy blocks the fill-from-empty case.

Note that Unpaywall cannot serve as a cross-check: since the OurResearch restructure it
**runs on OpenAlex data** as a child project. It inherits OpenAlex's errors. The genuine
independent cross-checks are arXiv ↔ Crossref ↔ S2.

## Filters

| queue | source | filter | max replicas | why |
|---|---|---|---|---|
| `harvest_arxiv` | arxiv | `{"any_of":["arxiv_id"]}` | 1 | 1 req / 3 s |
| `harvest_s2` | s2 | `{"any_of":["arxiv_id","doi"]}` | 2 | shared pool, frequent 429s |
| `harvest_openalex` | openalex | `{"any_of":["doi","openalex_id"]}` | 3 | metered since 2026-02-13 |

`maxReplicaCount` per queue **is** the rate-limit control. arXiv at 1 req/3s means one
pod and no shared token bucket — which is why `rate_budget` stays deferred until a source
genuinely needs more than one.

**`title` is deliberately not a filter value for any source.** Every API accepts a title,
so admitting it would re-create the unconditional fan-out — except each source would then
persist a *confidently wrong* record instead of an honest `not_found`. Title resolution
belongs in a resolver that verifies by **author-set overlap** (which caught 2/2 wrong
matches in testing) and is allowed to answer "no confident match".

Do not verify by year: OpenAlex dates the 2017 Attention paper to 2025, so a year check
would reject the *correct* record.

A typo'd field name is rejected at configuration time by a `CHECK` on
`harvest_filter_is_valid`. Evaluated lazily it would silently match nothing and the source
would simply never run — indistinguishable from "no seed has that identity yet", which is
the exact class of bug this design removes.

## Operations

```sql
-- what has never been asked, and why
select s.paper_id, s.arxiv_id is not null as has_arxiv, s.doi is not null as has_doi
  from seeds s
 where not exists (select 1 from harvest_dispatch d
                    where d.paper_id = s.paper_id and d.source = 'openalex');

-- re-run one source across the corpus
select public.harvest_republish('openalex', 'reason goes here');

-- current generations
select * from public.harvest_generation;
```

## Not in this design

- **The frontier** (references → new seeds). Recursion: 41 refs at 2 hops ≈ 1,700 papers,
  at 3 ≈ 70,000. Needs depth + relevance gating before it is wired.
- **Download.** A different consumer shape — identity → *artifact*, verified by sha256,
  via a tiered fallback that constructs URLs from the DOI prefix before trusting any API.
  See `graph-layout-rag/src/graph_layout_rag/harvest/doi_resolver.py`.
- **Seed merge.** `seeds` has `unique (corpus, doi)`. If seed A discovers a DOI seed B
  already holds, the update violates it — they are the same paper, seeded twice under
  different identities. Alias, merge, or reject is undecided.
- **Citations.** Unbounded (184,545 for one paper). Store `citationCount` as a scalar and
  derive `in_corpus_cited_by_count` by intersecting references within the corpus, as
  `citation_rank.py` does. Do not try to harvest the list.
