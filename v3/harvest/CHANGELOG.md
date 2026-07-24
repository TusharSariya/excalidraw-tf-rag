# Changelog

Notable changes to the `harvest` package. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions match `pyproject.toml`
(the image tag the `build-push` CI guard checks against prod ECR).

## [0.16.0]

### Fixed
- Stream S2 citation/reference pages into ≤1k-edge Writes **while paginating**
  instead of materializing the full related list on `S2Record` first. Prod
  pods (256Mi) were connection-reset / failing after fetching ~10k cites
  because RSS held the whole graph before upsert.
- Retry opening `/citations` and `/references` when the SDK raises TypeError
  on ``data: null`` (rate-limit flake). Mid-stream null pages abort before the
  outcome row so pgmq redelivers without a false `succeeded`.

## [0.15.0]

### Fixed
- Semantic Scholar **references were empty** even when `referenceCount > 0`:
  embedded `references` on paper/batch payloads often returns `[]` while the
  count is set (live example: `DOI:10.1111/cgf.12791` → 171 refs, 0 edges).
  Subject fetch is now **metadata-only** (`SUBJECT_FIELDS`: SEARCH + tldr +
  embedding); citations and references load via paginated
  `get_paper_citations` / `get_paper_references` (`limit=1000`).
- Large cite graphs no longer ship as one PostgREST upsert: `s2_writes` emits
  subject → ≤1000-edge chunks → `source_semantic_scholar` outcome last; the
  worker loops `list[Write]`. Cuts statement-timeout / OOM pressure on 256Mi
  pods (cluster memory bump remains infra-side if needed).

## [0.14.0]

### Fixed
- `SemanticScholarSource.get_papers`: stop requesting `Paper.FIELDS` on multi-id
  `POST /paper/batch`. Full nested cite/ref payloads make S2 return HTTP 400
  ("Response would exceed maximum size"); the SDK raises
  `BadQueryParametersException`, and we were treating that as all-not-found —
  so a 145-seed drain archived every row as `not_found` with empty `s2_*`
  tables. Fetch **one id at a time** with a **rich → thin → meta** field ladder
  (`RICH_FIELDS` ≈ RELATED_PAPER-shaped cites/refs + embedding/tldr; fall back
  when a mega-cited paper like Attention still overflows; meta keeps subject +
  embedding). Multi-id batches also silently zeroed `citations` for most papers
  even on HTTP 200. `s2` `batch_size` 500→1. Only map
  `BadQueryParametersException` to not-found for the genuine "no valid paper
  ids" case.

## [0.13.0]

### Added
- **12 new harvest sources**, each following the existing per-source pattern
  (typed record + tolerant `from_dict`, `PaperSource.get_papers`, `TableRow` +
  `<name>_write`, a `source_<name>` migration + `pgmq.create` + a capability-gated
  `queue_subscriptions` row): `unpaywall`, `crossref`, `dblp`, `openreview`,
  `acl_anthology`, `opencitations`, `datacite`, `core`, `internet_archive`,
  `zenodo`, `figshare`, `osf`. Sources with a native multi-id/filter endpoint
  (crossref `/works?filter=doi:…`, others) batch-fetch chunked to the API's id
  cap rather than looping singles; the rest keep the `get_papers(list)->dict`
  contract with a serial loop where no batch endpoint exists.
- **4 new seed identity columns** on `public.seeds` — `dblp_id`, `openreview_id`,
  `anthology_id`, `ia_identifier` — wired into every identity-enumeration site
  (`harvest_filter_is_valid`, `seed_satisfies`, `seeds_identity_monotonic`, the
  `seeds_identity_changed` trigger, and the Python `load_seed(s)` select lists).
- **Local live-stack test loop** — `v3/supabase/tests/local_harvest_loop.py` plus
  the local-only `pgmq_public` shim (`v3/supabase/tests/local_pgmq_public.sql`,
  never shipped as a migration — prod has it from the dashboard Queues
  integration) and a hand-authored smoke corpus
  (`v3/seed/seeds/harvest-smoke/popular.toml`) that populates every identity type.

### Notes
- `opencitations` caps citation/reference edges at **2000 per direction** (streamed,
  stops downloading at the cap) and uses a 30s read timeout. OpenCitations has no
  working server-side pagination and 503s on the largest citation streams (e.g.
  ResNet), so an unbounded fetch would hang the batch; the cap persists a partial
  neighborhood as `succeeded`, and a truly unservable DOI fails fast → pgmq
  redelivery → poison-pill backstop. `opencitations` runs at `batch_size=1` so one
  slow DOI cannot poison a batch of others.

### Config
- `HarvestSettings`: `unpaywall_email`, `crossref_mailto`, `core_api_key`.
  `UNPAYWALL_EMAIL` and `CORE_API_KEY` are required by their sources; the crossref
  mailto is the optional polite-pool hint.

### Not implemented (deliberately)
- Anna's Archive, Sci-Hub, Z-Library, and LibGen were **not** built: they
  distribute copyrighted works without authorization, which an automated
  corpus-scale harvester cannot use lawfully. Legal OA coverage comes from
  Unpaywall + CORE + Internet Archive + the Zenodo/Figshare/OSF deposit sources.

## [0.12.0]

### Fixed
- `SemanticScholarSource.get_papers`: pass `fields=Paper.FIELDS` so batch
  responses include `citations`, `references`, `embedding`, and `tldr`. The SDK
  defaults batch (`POST /paper/batch`) to `Paper.SEARCH_FIELDS`, which omits
  those — so `s2_citation` / `s2_reference` / `s2_embedding` stayed empty even
  when papers/authors wrote successfully.

## [0.11.0]

### Fixed
- `ArxivSource`: configure `arxiv.Client(delay_seconds=3, num_retries=5)` for ToU
  pacing. Live `0.10.0` workers were failing entire batches on `HTTP 429` from
  `export.arxiv.org` with nothing written to `source_arxiv`. Sustained 429s are
  left to pgmq redelivery rather than an in-process backoff: redelivery already
  spaces attempts by `VT_SECONDS` (120s), and sleeping in-process risked
  overrunning the visibility timeout, letting a second worker pick up the same
  batch mid-retry and inflating `read_ct` toward the `MAX_READ_CT` poison-pill
  cap on timing alone. A fetch now blocks for at most ~18s.
- `SupabasePersistence.read_batch`: stop waiting after **10s of continuous
  `visible_length == 0`** (drain-then-exit instead of hammering PostgREST for the
  full `WAIT_SECONDS`). Poll pacing is **0.1s for the first 1s**, then **1s**
  thereafter (empty and partial waits).

## [0.10.0]

### Fixed
- Batch worker loop: keep per-message `paper_id` ↔ `msg_id` mapping; poison-pill,
  not-found, write, and archive are all per-seed again. Each seed contributes a
  single fetch id — the first present identifier in `identify` order (e.g. arXiv
  before DOI) — so the deduped batch can never exceed a source's per-request id
  cap (S2 rejects batches over 500).
- `SupabasePersistence.read_batch` reads via `pgmq_public` with a real visibility
  timeout (replacing the broken `batch_load_seeds` that coerced pgmq rows into
  `Seed`). `load_seeds` returns `dict[int, Seed]` keyed by `paper_id`.
- `ArxivSource` / `OpenAlexSource` / `SemanticScholarSource.get_papers` key results
  by the requested identifier (and do not mutate the input list). OpenAlex batches
  W… ids; DOIs are resolved one-at-a-time because `Works()[list]` only filters by
  openalex id.

## [0.9.0]

### Added
- OpenAlex persistence: `harvest.persistence.models.openalex` maps a fetched
  `OpenAlexWork` to a `source_openalex` upsert (`SourceOpenAlexRow` +
  `openalex_write`), wired into `harvest-work` as the openalex source's
  `build_write`. The queryable subset is promoted to typed columns; the full Work
  is kept on `raw`, with `authorships` broken out to jsonb. Single row group — the
  citation graph (`referenced_works`) stays in `raw`, unlike `s2_write`'s fan-out.
- `sources.models.openalex.OpenAlexWork`: a typed, tolerant reader over the
  schemaless pyalex `Work` dict, with `abstract` de-inversion and a guarded
  `best_oa_pdf_url`.
- OpenAlex worker wiring resolves `openalex_id` first, then `doi`; `_works_key`
  normalises a bare `10.x` DOI to its `https://doi.org/…` URL form (a bare DOI
  404s on `/works/`) and namespaces ROR ids (inert — ROR is institution-scoped).
- Migration `20260720120000_source_openalex_enrich.sql` promotes
  `publication_year`, `language`, `is_oa`, `oa_status`, `referenced_works_count`,
  and `is_retracted` to typed `source_openalex` columns. **Must be applied to
  staging before the first openalex write.**

### Changed
- `sources.base` reduced to a generic `PaperSource` Protocol; each source returns
  its own rich record (`arxiv.Result` / `S2Record` / `OpenAlexWork`) rather than a
  shared `Paper`/`Author` dataclass. `SemanticScholarSource.get_paper` now returns
  `S2Record`.

## [0.8.0]

### Added
- arXiv persistence: `harvest.persistence.models.arxiv` maps a fetched
  `arxiv.Result` to a `source_arxiv` upsert (`SourceArxivRow` + `arxiv_write`),
  wired into `harvest-work` as the arxiv source's `build_write`. Nested
  `Author`/`Link` objects are projected to jsonb (affiliation and link labels
  kept), alongside a faithful `raw` projection. No migration — `source_arxiv`
  was already lossless (`20260716200000`).

### Fixed
- arXiv worker no longer crashes on persist. The arxiv source was wired to
  `s2_write`, which requires an S2 record's `paper_id`; it raised
  `AttributeError` on every fetched paper and left the message for redelivery
  until the poison cap marked it error. It now uses the arxiv-specific writer.
- `ArxivSource.get_paper` returns `arxiv.Result` directly and maps a genuine
  not-found (`StopIteration` on an empty first page) to `None`; transient
  failures (`HTTPError`/`ConnectionError`, after the client's own `num_retries`)
  propagate so the worker leaves the message for pgmq redelivery rather than
  mislabelling a live paper not-found.
- `SupabaseSettings` loads `.env` with `extra="ignore"` and without
  `cli_parse_args`, so instantiating it outside the entrypoint no longer tries
  to re-parse `--source`.

### Changed
- Renamed `PaperRecord` → `S2Record` (the Semantic Scholar record DTO) across
  sources, persistence, and tests — it is S2-specific, not a generic record.

### Known gaps
- `openalex` is still wired to `s2_write` and will raise the same way arxiv did
  once a seed resolves an OpenAlex id; it needs its own `SourceOpenAlexRow` +
  `openalex_write`.

## [0.7.0]

### Added
- Semantic Scholar persistence (`harvest.persistence`): a `PaperRecord` maps to
  an FK-ordered multi-table upsert over the new `s2_*` graph tables
  (paper/author/authorship/reference/citation/embedding) plus the per-seed
  `source_semantic_scholar` outcome row, written together per fetched paper.
  Migration `20260718212350_s2_graph_tables.sql`.

### Fixed
- `SemanticScholarSource.get_paper` sends S2-namespaced ids (`ARXIV:`/`DOI:`)
  to the Graph API verbatim instead of calling a nonexistent `_schpaper_id`
  helper (which would have raised on first use). It now swallows only a genuine
  not-found to `None`; transient errors (rate limit, network) propagate so the
  worker leaves the message for pgmq redelivery rather than mislabelling the
  seed not-found.
- `harvest-work` tries each candidate id in order (for s2: arXiv, then DOI),
  stopping at the first that resolves; `mark_not_found` fires only when every
  candidate is genuinely absent.

### Removed
- Legacy one-shot `harvest` CLI (`harvest:main` / PDF text extraction via
  pdfplumber). The image and console entrypoint are now `harvest-work` only
  (`ENTRYPOINT ["harvest-work"]`; pass `--source`). Dropped the unused
  `pdfplumber` dependency.

## [0.6.0]

### Added
- `OpenAlexSource` — the third `PaperSource`, backed by the **pyalex** SDK (OpenAlex's
  own recommended Python client) rather than raw REST calls. Completes the source set
  the queues have fanned out to since `20260710235341`; `harvest_openalex` messages
  have been draining as no-ops until now.
  - Unlike arXiv, OpenAlex has a citation graph, so `get_references` / `get_citations`
    return real data. References resolve in batches (`openalex:W1|W2|…`) rather than
    one lookup per reference.
  - Retries 429/5xx (pyalex defaults to `max_retries=0`, which would turn a transient
    rate-limit into a hard failure indistinguishable from a real fetch error).
  - The API key is optional but raises the free allowance from ~$0.10/day to ~$1/day;
    read from `OPENALEX_API_KEY`, and its absence is logged rather than assumed.

### Fixed
- **`ArxivSource.get_authors()` raised `AttributeError` on every call.** `_fetch` is
  annotated `-> arxiv.Result | None` but actually returns an `ArxivPaper`, whose
  `authors` was a `list[str]` — so `author.name` ran against a plain string. Nothing
  caught it because no handler calls `get_authors` yet; it would have surfaced the
  moment the per-source handler landed.
- Removed unreachable code in `ArxivSource._fetch` (a `logger.info` + `return paper`
  after the `except` blocks, which the `try` can never fall through to) and the
  per-paper `pprint` dump left over from debugging.

### Changed
- **`ArxivPaper` no longer flattens nested library objects** — it was losing data on
  the way in, which is the one thing a harvester must not do, since re-fetching is the
  expensive, rate-limited step:
  - `authors: list[str]` → `list[dict]`. `arxiv.Author` carries `affiliation` as well
    as `name`; a name-only list dropped it.
  - `links: list[str]` → `list[dict]`. `arxiv.Result.Link` carries `title`, `rel` and
    `content_type` — the labels identifying *which* link is the PDF. Bare hrefs force
    string-matching `arxiv.org/pdf` to find it back.
  - New `as_raw()` for the `source_arxiv.raw` column. Note it is a faithful projection
    of `arxiv.Result`, not the Atom XML: the client keeps no `_raw`, unlike the
    OpenAlex/S2 SDKs which hand back dicts.
  - Paired with migration `20260716200000_source_arxiv_lossless.sql`. Safe to reshape
    in place — all three `source_*` tables are empty, since no handler writes yet.

### Notes
- **`OpenAlexSource`'s `identifier` is a DOI or OpenAlex id — NOT an arXiv id**, unlike
  the other two sources. This is forced, not chosen: OpenAlex has no arXiv filter, and
  arXiv's own DOI (`10.48550/arXiv.<id>`) does not resolve (404). Title search is not a
  usable fallback — searching "Attention Is All You Need" returns a `type: preprint`
  dated 2025 whose PDF sits on a scraping mirror. A harvester would upsert that
  happily, so the source returns `None` rather than guessing.
- `public.seeds` already carries `openalex_id` and `doi` columns, and the planned
  `get_identity()` reads `arxiv_id`/`doi` — so the caller passes the identity that fits
  the source. A seed with only an `arxiv_id` cannot be harvested from OpenAlex without
  first resolving a DOI elsewhere (Semantic Scholar's `externalIds` carries one).

## [0.5.0]

### Added
- `harvest-work` worker — a single-threaded, drain-then-exit pull loop over the
  Supabase `pgmq_public` queue wrappers (`read` → `contracts.parse` → `archive`,
  exit when the queue empties). Proven end-to-end against staging.
- Per-entrypoint settings: a shared base (`SUPABASE_URL` / `SUPABASE_SERVICE_KEY`
  + source API keys, with a fail-fast `client()`) plus `WorkerSettings`
  (`--source`) and `SeederSettings` (`--corpus`/`--arxiv_id`/`--doi`).

### Changed
- README rewritten for the seed → queue → worker architecture.
- Settings drop the legacy REST url/key and the direct-pg DSN in favour of
  `supabase_url` / `supabase_service_key`.

## [0.4.0]

### Added
- Seed → message contract in `contracts/`: a versioned JSON Schema
  (`seed_message.v1.json`), a **generated** `SeedMessage` pydantic model, and a
  tolerant-reader `parse()` — additive fields warn + are ignored;
  subtractive/mistyped/unrecognized-version messages raise.
- DB enforcement (migration `20260711040000`): `publish_seed` stamps
  `{"v":1,…}` and `pg_jsonschema`-validates every message before enqueuing.

## [0.3.0]

- Baseline: legacy one-shot harvester (arXiv + Semantic Scholar fetch over the
  Supabase REST client). Superseded by the decoupled seed / queue / worker design.
