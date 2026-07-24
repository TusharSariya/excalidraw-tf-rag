-- ACL Anthology harvest source. Per-seed OUTCOME table + its queue + subscription.
-- Mirrors source_unpaywall (single-table, single-key source): a row is created
-- (status='pending') when the source is attempted for a seed, flips to 'succeeded'
-- with the typed data filled in, or 'not_found'/'error'. Every data column is
-- nullable — it's populated only on success. paper_id is a bigint FK to seeds;
-- on delete cascade drops a seed's outcome with the seed.
--
-- ACL Anthology is keyed on its own paper id (anthology_id, e.g. N19-1423) — hence
-- filter {"any_of":["anthology_id"]}. anthology_id became a recognised identity
-- field in 20260722100000_seed_identity_columns.sql (seeds column + filter/dispatch
-- wiring), so this filter passes harvest_filter_is_valid.

-- source_acl_anthology — typed for the fields we use now; raw jsonb keeps the whole
-- parsed BibTeX entry (plus the real raw_bibtex bytes).
create table public.source_acl_anthology (
    paper_id       bigint primary key references public.seeds (paper_id) on delete cascade,
    status         source_status not null default 'pending',
    anthology_id   text,
    title          text,
    authors        text[],        -- 'Last, First' names, split from the bib author string
    year           integer,
    venue          text,          -- booktitle (inproceedings) or journal (@article)
    doi            text,          -- optional; not present on every record
    url            text,          -- canonical landing page
    pdf_url        text,          -- derived from url (…/N19-1423/ -> …/N19-1423.pdf)
    bibkey         text,          -- BibTeX cite key (devlin-etal-2019-bert)
    entry_type     text,          -- inproceedings / article / …
    raw            jsonb,
    error          text,          -- populated when status = 'error'
    fetched_at     timestamptz    -- set when status = 'succeeded'
);
alter table public.source_acl_anthology enable row level security;

-- Queue: creates pgmq.q_harvest_acl_anthology (live) + pgmq.a_harvest_acl_anthology (archive).
select pgmq.create('harvest_acl_anthology');

-- Subscribe the source. filter {"any_of":["anthology_id"]}: dispatched only to seeds
-- that carry an anthology_id (ACL Anthology has no other lookup key). Adding a source
-- is an INSERT.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_acl_anthology', 'acl_anthology', '{"any_of": ["anthology_id"]}'::jsonb);
