-- source_datacite — per-seed OUTCOME of the DataCite fetch, keyed by seeds.paper_id.
-- Mirrors source_openalex: type the queryable subset, keep the whole JSON:API record
-- on `raw`. DataCite registers dataset/software DOIs (Zenodo, Dryad, figshare, …);
-- most journal/paper DOIs live at Crossref and legitimately 404 here (a valid
-- not_found, recorded as status='not_found' by the worker — not an error).
--
-- Every data column is nullable: populated only on a successful fetch. on delete
-- cascade — dropping a seed drops its outcome. The `doi` seed identity satisfies the
-- existing queue_subscriptions filter check (harvest_filter_is_valid allows 'doi').

create table public.source_datacite (
    paper_id               bigint primary key references public.seeds (paper_id) on delete cascade,
    status                 source_status not null default 'pending',
    doi                    text,
    prefix                 text,
    suffix                 text,
    state                  text,
    title                  text,
    publisher              text,
    publication_year       integer,
    resource_type_general  text,
    resource_type          text,
    url                    text,
    version                text,
    language               text,
    schema_version         text,
    citation_count         integer,
    reference_count        integer,
    is_active              boolean,
    registered             timestamptz,
    created                timestamptz,
    updated                timestamptz,
    creators               jsonb,
    raw                    jsonb,
    error                  text,          -- populated when status = 'error'
    fetched_at             timestamptz    -- set when status = 'succeeded'
);
alter table public.source_datacite enable row level security;

comment on column public.source_datacite.resource_type_general is
    'DataCite types.resourceTypeGeneral — Software / Dataset / Text / ….';
comment on column public.source_datacite.state is
    'DataCite attributes.state — e.g. ''findable''.';

-- One queue for the source. Creates pgmq.q_harvest_datacite (live) + a_… (archive).
select pgmq.create('harvest_datacite');

-- Register the subscription. filter = the identity the source can work from: a DOI.
-- 'doi' passes public.harvest_filter_is_valid, so the queue_subscriptions_filter_valid
-- check constraint accepts it.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_datacite', 'datacite', '{"any_of": ["doi"]}'::jsonb);
