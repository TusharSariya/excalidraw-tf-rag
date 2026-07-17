-- Make source_arxiv lossless, matching source_openalex and source_semantic_scholar.
--
-- Those two carry `raw jsonb`, so the whole API payload survives and anything we did
-- not think to type is still recoverable. source_arxiv had no such column, and two of
-- its typed columns flattened nested data on the way in — silently, at write time:
--
--   * authors text[]  <- [a.name for a in result.authors]
--     arxiv.Author carries `affiliation` as well as `name`. Affiliation was dropped.
--
--   * links text[]    <- [l.href for l in result.links]
--     arxiv.Result.Link carries href, title, rel AND content_type. So we kept the URLs
--     but threw away the labels that say WHICH url is which — title='pdf',
--     rel='related', content_type='application/pdf'. Reconstructing "the PDF link"
--     from a bare href list means string-matching arxiv.org/pdf, which is exactly the
--     kind of guess this table exists to avoid.
--
-- Harvest is the expensive, rate-limited, network-bound step; re-fetching to recover a
-- field we chose not to store is the one cost worth designing against. Persist
-- faithfully, reshape later.
--
-- Safe to change shape in place: all three source_* tables are EMPTY (verified
-- 2026-07-16) because the worker's per-source handler is still unwritten. There is
-- nothing to backfill, and this is the cheapest this fix will ever be.
--
-- Caveat on `raw`: the arxiv library does not retain the source Atom XML (there is no
-- Result._raw), so `raw` here is a faithful projection of arxiv.Result — every public
-- attribute, with nested Authors/Links kept whole — rather than the literal API
-- response. That is the most fidelity the client makes available; OpenAlex/S2 `raw` are
-- true payloads because those SDKs return dicts.

-- authors: [{"name": ..., "affiliation": ...}, ...]
alter table public.source_arxiv drop column authors;
alter table public.source_arxiv add column authors jsonb not null default '[]'::jsonb;

-- links: [{"href": ..., "title": ..., "rel": ..., "content_type": ...}, ...]
alter table public.source_arxiv drop column links;
alter table public.source_arxiv add column links jsonb not null default '[]'::jsonb;

-- The safety net the other two sources already had.
alter table public.source_arxiv add column raw jsonb;

comment on column public.source_arxiv.authors is
    'Ordered [{name, affiliation}] from arxiv.Result.authors. jsonb not text[]: Author '
    'carries affiliation, which a name-only array silently drops.';
comment on column public.source_arxiv.links is
    'Ordered [{href, title, rel, content_type}] from arxiv.Result.links. jsonb not '
    'text[]: the labels identify which link is the PDF; bare hrefs force string '
    'matching on arxiv.org/pdf.';
comment on column public.source_arxiv.raw is
    'Faithful projection of arxiv.Result (not the Atom XML — the client does not retain '
    'it). Mirrors source_openalex.raw / source_semantic_scholar.raw so no field is lost '
    'to a typing decision made before we knew what we needed.';
