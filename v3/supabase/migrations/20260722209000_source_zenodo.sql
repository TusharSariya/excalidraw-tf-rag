-- Zenodo harvest source. Per-seed OUTCOME table + its queue + subscription.
-- Mirrors source_unpaywall / source_openalex (single-table, DOI-keyed source): a
-- row is created (status='pending') when the source is attempted for a seed, flips
-- to 'succeeded' with the typed data filled in, or 'not_found'/'error'. Every data
-- column is nullable — it's populated only on success. paper_id is a bigint FK to
-- seeds; on delete cascade drops a seed's outcome with the seed.
--
-- Zenodo is reached by DOI only (GET /api/records?q=doi:"{doi}" OR conceptdoi:"{doi}")
-- — hence filter {"any_of":["doi"]}. Many paper DOIs are not Zenodo-hosted; those
-- resolve to status='not_found' (hits.total == 0), which is not an error.

-- source_zenodo — typed for the fields we use now; raw jsonb keeps the rest.
create table public.source_zenodo (
    paper_id            bigint primary key references public.seeds (paper_id) on delete cascade,
    status              source_status not null default 'pending',
    record_id           bigint,        -- Zenodo numeric record id (version-specific)
    doi                 text,          -- version DOI of this record
    conceptdoi          text,          -- concept (all-versions) DOI
    doi_url             text,
    title               text,
    publication_date    date,
    access_right        text,          -- e.g. 'open'
    version             text,          -- e.g. 'v3.7.2'
    resource_type_type  text,          -- e.g. 'software'
    license_id          text,          -- e.g. 'cc-by-4.0'
    file_download_url   text,          -- first deposited file's direct content URL
    landing_page        text,          -- human record page (links.self_html)
    creators            jsonb,         -- metadata.creators, broken out of raw for querying
    files               jsonb,         -- deposit files array, broken out of raw for querying
    raw                 jsonb,
    error               text,          -- populated when status = 'error'
    fetched_at          timestamptz    -- set when status = 'succeeded'
);
alter table public.source_zenodo enable row level security;

-- Queue: creates pgmq.q_harvest_zenodo (live) + pgmq.a_harvest_zenodo (archive).
select pgmq.create('harvest_zenodo');

-- Subscribe the source. filter {"any_of":["doi"]}: dispatched only to seeds that
-- carry a DOI (Zenodo has no other lookup key). Adding a source is an INSERT.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_zenodo', 'zenodo', '{"any_of": ["doi"]}'::jsonb);
