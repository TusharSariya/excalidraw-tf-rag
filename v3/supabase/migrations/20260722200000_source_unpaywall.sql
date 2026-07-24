-- Unpaywall harvest source. Per-seed OUTCOME table + its queue + subscription.
-- Mirrors source_openalex (single-table, DOI-keyed source): a row is created
-- (status='pending') when the source is attempted for a seed, flips to
-- 'succeeded' with the typed data filled in, or 'not_found'/'error'. Every data
-- column is nullable — it's populated only on success. paper_id is a bigint FK
-- to seeds; on delete cascade drops a seed's outcome with the seed.
--
-- Unpaywall is keyed on DOI only (GET /v2/{doi}) — hence filter {"any_of":["doi"]}.

-- source_unpaywall — typed for the fields we use now; raw jsonb keeps the rest.
create table public.source_unpaywall (
    paper_id            bigint primary key references public.seeds (paper_id) on delete cascade,
    status              source_status not null default 'pending',
    doi                 text,
    is_oa               boolean,
    oa_status           text,          -- gold/green/bronze/hybrid/closed
    title               text,
    genre               text,
    published_date      date,
    year                integer,
    publisher           text,
    journal_name        text,
    url_for_pdf         text,          -- best_oa_location.url_for_pdf (primary PDF target)
    best_oa_url         text,          -- best_oa_location.url
    best_oa_host_type   text,          -- publisher/repository
    best_oa_version     text,          -- publishedVersion/acceptedVersion/submittedVersion
    best_oa_license     text,
    oa_locations        jsonb,         -- all OA copies, broken out of raw for querying
    raw                 jsonb,
    error               text,          -- populated when status = 'error'
    fetched_at          timestamptz    -- set when status = 'succeeded'
);
alter table public.source_unpaywall enable row level security;

-- Queue: creates pgmq.q_harvest_unpaywall (live) + pgmq.a_harvest_unpaywall (archive).
select pgmq.create('harvest_unpaywall');

-- Subscribe the source. filter {"any_of":["doi"]}: dispatched only to seeds that
-- carry a DOI (Unpaywall has no other lookup key). Adding a source is an INSERT.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_unpaywall', 'unpaywall', '{"any_of": ["doi"]}'::jsonb);
