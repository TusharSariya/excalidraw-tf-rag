-- Enrich source_openalex with the high-value typed columns the OpenAlexWork model
-- (v3/harvest sources/models/openalex.py) now exposes, so they are queryable without
-- digging into `raw`. Everything else stays in `raw` (already present) — the same
-- "type the useful, keep raw for the rest" posture as source_arxiv.
--
-- All columns are nullable and populated only on a successful fetch, so the table
-- needs no backfill: the per-source openalex handler has never persisted a success
-- (it was mis-wired to the S2 writer), so only pending/not_found rows exist.

alter table public.source_openalex
    add column if not exists publication_year       integer,
    add column if not exists language               text,
    add column if not exists is_oa                  boolean,
    add column if not exists oa_status              text,
    add column if not exists referenced_works_count integer,
    add column if not exists is_retracted           boolean;

comment on column public.source_openalex.publication_year is
    'OpenAlex Work.publication_year — the usual filter; publication_date keeps the full date.';
comment on column public.source_openalex.language is
    'Work.language — ISO 639-1 code, e.g. "en" (nullable; OpenAlex omits it for some works).';
comment on column public.source_openalex.is_oa is
    'Work.open_access.is_oa — whether an open-access copy exists.';
comment on column public.source_openalex.oa_status is
    'Work.open_access.oa_status — gold / green / hybrid / bronze / closed.';
comment on column public.source_openalex.referenced_works_count is
    'Work.referenced_works_count — size of the outbound citation set; the ids themselves '
    'live in raw->referenced_works (no per-openalex graph tables, unlike s2).';
comment on column public.source_openalex.is_retracted is 'Work.is_retracted.';
