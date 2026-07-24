-- source_core (type key 'core') — per-seed outcome of the CORE v3 fetch.
--
-- CORE resolves a work by DOI (bearer-auth only; no unauthenticated tier). The
-- point of the source is the open-access fulltext: core_id doubles as CORE's
-- download id, and download_url / full_text_identifier carry the OA fulltext URL.
-- Typed for the fields we query now; raw jsonb keeps the whole work response.
-- Every data column is nullable — populated only on status='succeeded'.
create table public.source_core (
    paper_id              bigint primary key references public.seeds (paper_id) on delete cascade,
    status                source_status not null default 'pending',
    core_id               text,          -- CORE numeric work id (= OA download id), kept as text
    doi                   text,
    title                 text,
    download_url          text,          -- OA fulltext download URL
    full_text_identifier  text,          -- OA fulltext URL / identifier
    year_published        integer,
    published_date        text,
    publisher             text,
    language              text,          -- language.code
    document_type         text,
    data_providers        jsonb,
    raw                   jsonb,
    error                 text,          -- populated when status = 'error'
    fetched_at            timestamptz    -- set when status = 'succeeded'
);
alter table public.source_core enable row level security;

-- One queue for the source: creates pgmq.q_harvest_core (live) + pgmq.a_harvest_core (archive).
select pgmq.create('harvest_core');

-- Subscribe: fan out to core only for seeds that carry a DOI (its only usable identity).
insert into public.queue_subscriptions (queue_name, source, filter)
values ('harvest_core', 'core', '{"any_of": ["doi"]}'::jsonb);
