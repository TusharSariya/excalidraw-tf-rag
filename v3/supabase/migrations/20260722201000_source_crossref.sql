-- Crossref source (type key 'crossref'). Two tables:
--   source_crossref     — per-seed OUTCOME (mirrors the other source_* tables):
--                         pending -> succeeded (typed cols filled) / not_found / error.
--                         raw jsonb keeps the full Crossref item; typed cols are the
--                         queryable subset. DOI is stored as Crossref returns it
--                         (LOWERCASED), so case-fold when joining to seeds.doi.
--   crossref_reference  — outbound reference edges fanned out from message.items[].reference[].
--                         Each edge FKs into seeds (there is no Crossref node table);
--                         target_doi is set only when the reference carries a DOI
--                         (~30-70% of entries), else the reference is DOI-less/unstructured.

-- source_crossref — typed for the fields we use now; raw jsonb keeps the rest.
create table public.source_crossref (
    paper_id                bigint primary key references public.seeds (paper_id) on delete cascade,
    status                  source_status not null default 'pending',
    doi                     text,
    title                   text,
    container_title         text,
    type                    text,
    publisher               text,
    url                     text,
    issued_year             integer,
    is_referenced_by_count  integer,
    references_count        integer,
    page                    text,
    issn                    text[],
    authors                 jsonb,
    raw                     jsonb,
    error                   text,
    fetched_at              timestamptz
);
alter table public.source_crossref enable row level security;

comment on table public.source_crossref is
    'Crossref per-seed harvest outcome. Keyed by seeds.paper_id. DOI stored lowercased.';

-- crossref_reference — outbound reference edges (subject seed -> reference entry).
-- ref_key is Crossref's per-work reference[].key, always present and unique within a
-- work; it is the edge's natural key alongside the seed's paper_id.
create table public.crossref_reference (
    paper_id         bigint not null references public.seeds (paper_id) on delete cascade,
    ref_key          text not null,
    target_doi       text,
    doi_asserted_by  text,
    article_title    text,
    journal_title    text,
    author           text,
    year             text,
    volume           text,
    first_page       text,
    unstructured     text,
    raw              jsonb,
    primary key (paper_id, ref_key)
);
create index crossref_reference_target_doi_idx on public.crossref_reference (target_doi);
alter table public.crossref_reference enable row level security;

comment on table public.crossref_reference is
    'Outbound Crossref references: seed paper_id -> reference entry. target_doi set only '
    'when the reference carries a DOI (~30-70% of entries); else DOI-less/unstructured.';

-- One queue for the source (pgmq.q_harvest_crossref live + a_harvest_crossref archive).
select pgmq.create('harvest_crossref');

-- Subscribe: Crossref resolves from a DOI. The filter gate means a seed is dispatched
-- to harvest_crossref only once it carries a doi.
insert into public.queue_subscriptions (queue_name, source, filter)
values ('harvest_crossref', 'crossref', '{"any_of": ["doi"]}'::jsonb);
