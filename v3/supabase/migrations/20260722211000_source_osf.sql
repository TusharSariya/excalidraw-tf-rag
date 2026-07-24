-- OSF preprint harvest source. Per-seed OUTCOME table + its queue + subscription.
-- Mirrors source_unpaywall (single-table, DOI-keyed source): a row is created
-- (status='pending') when the source is attempted for a seed, flips to
-- 'succeeded' with the typed data filled in, or 'not_found'/'error'. Every data
-- column is nullable — it's populated only on success. paper_id is a bigint FK
-- to seeds; on delete cascade drops a seed's outcome with the seed.
--
-- Keyed on the seed's DOI (filter {"any_of":["doi"]}), but OSF's preprints
-- endpoint does NOT support filter[doi] and has no external-DOI lookup — the
-- source resolves the DOI to an OSF-native guid (an OSF-minted DOI
-- 10.31234/osf.io/<guid>, or a bare guid) and records not_found otherwise. The
-- meaningful OSF DOI lives at data.links.preprint_doi; data.attributes.doi is
-- the published-article DOI and is null on most preprints.

-- source_osf — typed for the fields we use now; raw jsonb keeps the rest.
create table public.source_osf (
    paper_id          bigint primary key references public.seeds (paper_id) on delete cascade,
    status            source_status not null default 'pending',
    osf_id            text,          -- OSF preprint guid, e.g. 'k3fr9_v1'
    doi               text,          -- published-article DOI (attributes.doi; often null)
    preprint_doi      text,          -- OSF-minted DOI url (links.preprint_doi)
    title             text,
    description       text,          -- abstract text
    date_published    timestamptz,
    date_modified     timestamptz,
    is_published      boolean,
    reviews_state     text,          -- e.g. 'accepted'
    provider_id       text,          -- e.g. 'psyarxiv'
    node_id           text,          -- backing OSF project guid
    html_url          text,          -- landing page (links.html)
    primary_file_id   text,
    primary_file_url  text,          -- api.osf.io/v2/files/{id}/ (file metadata + download)
    files_url         text,          -- listing of all attached files
    tags              text[],
    subjects          jsonb,         -- nested taxonomy paths, broken out of raw
    license_record    jsonb,         -- {copyright_holders, year}, broken out of raw
    raw               jsonb,
    error             text,          -- populated when status = 'error'
    fetched_at        timestamptz    -- set when status = 'succeeded'
);
alter table public.source_osf enable row level security;

-- Queue: creates pgmq.q_harvest_osf (live) + pgmq.a_harvest_osf (archive).
select pgmq.create('harvest_osf');

-- Subscribe the source. filter {"any_of":["doi"]}: dispatched only to seeds that
-- carry a DOI (OSF's only seed key here). Adding a source is an INSERT.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_osf', 'osf', '{"any_of": ["doi"]}'::jsonb);
