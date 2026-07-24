-- OpenCitations harvest source (type key 'opencitations'). A fan-out source, like
-- s2: a DOI->DOI citation graph (opencitations_edge) plus a per-seed OUTCOME table
-- (source_opencitations) + its queue + subscription.
--
-- OpenCitations' Index v2 is DOI-addressable via the `doi:` id form (bare {doi}
-- 404s). GET /index/v2/references/doi:{doi} yields the works the seed CITES; GET
-- /index/v2/citations/doi:{doi} yields the works that CITE the seed. Hence the
-- filter {"any_of":["doi"]} — the seed's only usable identity here. No API key
-- required.

-- opencitations_edge — one row per citation (citing_doi cites cited_doi).
-- Keyed by `oci`, OpenCitations' stable, globally-unique edge id: two seeds can
-- surface the same edge (one via its references, the other via its citations) and
-- both upsert this one row, so the edge is NOT tied to a seed. citing_doi/cited_doi
-- are bare lowercased DOIs parsed out of the space-separated `citing`/`cited`
-- bundles; the whole element (incl. the omid/openalex/pmid ids the typed columns
-- drop) is kept on raw. This is where the full API response is preserved.
create table public.opencitations_edge (
    oci          text primary key,
    citing_doi   text not null,
    cited_doi    text not null,
    creation     text,          -- ISO-ish date of the citing work; may be a bare year
    timespan     text,          -- ISO-8601 duration between cited and citing; may be negative
    journal_sc   boolean,       -- journal self-citation flag
    author_sc    boolean,       -- author self-citation flag
    raw          jsonb,         -- the whole citation element (preserves the bundle ids)
    fetched_at   timestamptz
);
create index opencitations_edge_citing_idx on public.opencitations_edge (citing_doi);
create index opencitations_edge_cited_idx on public.opencitations_edge (cited_doi);
alter table public.opencitations_edge enable row level security;

comment on table public.opencitations_edge is
    'OpenCitations DOI->DOI citation edges. Natural key = oci. Query outgoing refs of '
    'DOI D with where citing_doi = D; incoming citations with where cited_doi = D.';

-- source_opencitations — per-seed outcome. A row is created (status='pending') when
-- the source is attempted for a seed, flips to 'succeeded' with the counts filled
-- in, or 'not_found'/'error'. Every data column is nullable — populated only on
-- success. paper_id is a bigint FK to seeds; on delete cascade drops the outcome
-- with the seed. raw stays null: full fidelity lives on opencitations_edge.raw.
create table public.source_opencitations (
    paper_id          bigint primary key references public.seeds (paper_id) on delete cascade,
    status            source_status not null default 'pending',
    doi               text,
    reference_count   integer,       -- elements from /references/doi:{seed}
    citation_count    integer,       -- elements from /citations/doi:{seed}
    raw               jsonb,
    error             text,          -- populated when status = 'error'
    fetched_at        timestamptz    -- set when status = 'succeeded'
);
alter table public.source_opencitations enable row level security;

-- Queue: creates pgmq.q_harvest_opencitations (live) + pgmq.a_harvest_opencitations (archive).
select pgmq.create('harvest_opencitations');

-- Subscribe the source. filter {"any_of":["doi"]}: dispatched only to seeds that
-- carry a DOI (OpenCitations has no other lookup key). Adding a source is an INSERT.
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_opencitations', 'opencitations', '{"any_of": ["doi"]}'::jsonb);
