# v3/seed

A local, TOML-driven bulk seeder for `public.seeds`. Reads papers from a TOML file
or a directory of them (the `seeds/<corpus>/<theme>.toml` catalog), inserts each into
the `seeds` table, and publishes it to the active harvest queues.

This is the bulk successor to `v3/supabase/scripts/seed.py` (which seeds one paper
per invocation via flags), and does what the documented-but-unbuilt `harvest-seed`
entrypoint is specified to do. It is a standalone `uv run` script — no package,
lockfile, or image.

## Usage

`seed.py` takes a single TOML file **or a directory** of them. The seed catalog
lives under `seeds/<corpus>/<theme>.toml`, so you seed a whole corpus, everything,
or a single theme file:

```bash
uv run v3/seed/seed.py                                # everything under ./seeds (all corpora)
uv run v3/seed/seed.py seeds/graph-layout-rag         # one corpus (all its theme files)
uv run v3/seed/seed.py seeds/rag-literature-rag/graphrag.toml   # a single theme file
uv run v3/seed/seed.py --dry-run                      # validate + per-corpus counts, no DB writes
uv run v3/seed/seed.py --corpus my-corpus             # override the corpus for every seed
uv run v3/seed/seed.py --no-publish                   # insert only; skip the publish_seed RPC
```

When given a directory it loads every `*.toml` beneath it (recursively); each file
carries its own top-level `corpus`, so one run can seed several corpora at once.

## Seed catalog layout

```
seeds/
  graph-layout-rag/        # corpus = "graph-layout-rag"
    sugiyama-pipeline.toml  force-directed-stress.toml  routing-edges.toml
    constraints-compound.toml  compaction-packing.toml  trees-planar-geometry.toml
    analytics.toml  neural-graph-drawing.toml  foundational-books.toml  uncategorized.toml
  rag-literature-rag/      # corpus = "rag-literature-rag"
    foundations.toml  dense-retrieval-embeddings.toml  late-interaction-sparse.toml
    reranking.toml  graphrag.toml  agentic.toml  self-correcting.toml
    chunking-indexing.toml  query-transformation.toml  long-context.toml
    evaluation.toml  training-memory-compression.toml  uncategorized.toml
```

One directory per corpus, one file per research theme, every seed carrying a human
title. To **add a seed**, append a `[[seed]]` to the right theme file. To **add a
corpus**, create a new `seeds/<corpus>/` subdirectory.

### Regenerating from the RAG tools (`build_seeds.py`)

The catalog was bootstrapped from `graph-layout-rag` / `rag-literature-rag`'s harvest
modules by `build_seeds.py`, which extracts every DOI / arxiv identity, resolves a
title (Crossref → OpenAlex, cached in `.title_cache.json`), classifies it by theme,
and writes the files. Re-run it when an upstream catalog grows:

```bash
uv run v3/seed/build_seeds.py               # rebuild both corpora
uv run v3/seed/build_seeds.py --no-resolve  # skip network title lookups (fast)
```

The generated TOMLs are the source of truth and are hand-editable; the generator is a
convenience (its corpus specs + theme keyword maps live at the top of the script).

## TOML format

Each theme file has this shape — one top-level `corpus`, then a `[[seed]]` array:

```toml
corpus = "graph-layout-rag"        # default corpus for every seed below

[[seed]]
title = "ForceAtlas2, a Continuous Graph Layout Algorithm"
doi   = "10.1371/journal.pone.0098679"

[[seed]]
arxiv_id = "2311.00533"
corpus   = "some-other-corpus"     # optional per-seed override
```

Each `[[seed]]` needs a `corpus` (default, per-seed, or `--corpus`) and at least one
identity: `doi`, `arxiv_id`, `openalex_id`, or `title`. Only `arxiv_id` seeds exercise
the arxiv source; `doi`-only seeds resolve via Semantic Scholar (and OpenAlex once
wired), with arxiv marking them `not_found` by design.

## Environment

Needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (the `service_role` secret — the
`seeds` table is RLS-locked to the backend). Taken from the environment, or, if
unset, read from `../harvest/.env`.

## Behaviour

- **Non-destructive.** It only ever inserts. Re-seeding a paper already present in a
  corpus reuses its `paper_id` (dedup on the `(corpus, doi|arxiv_id|openalex_id)`
  unique constraints) and re-publishes it — safe to re-run against staging.
  A *title-only* seed has no unique constraint to dedup on, so re-running one will
  insert a duplicate row.
- **Fan-out.** Inserting a row fires the `seeds` `AFTER INSERT` trigger, which calls
  `publish_seed`. The script also calls `publish_seed` explicitly (idempotent —
  dispatch is claimed per generation); that explicit call is what re-enqueues an
  already-existing seed. Pass `--no-publish` to insert without the explicit call.

To purge seeded data from staging afterwards, use
`v3/supabase/scripts/purge_staging.sh`.
