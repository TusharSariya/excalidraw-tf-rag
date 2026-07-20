-- Semantic Scholar graph tables (type key 's2').
-- Separate from source_semantic_scholar (harvest outcome keyed by seeds.paper_id).
-- Citations and references are separate edge tables so a highly cited paper's
-- fan-in only grows s2_citation and does not bloat reference scans.
-- 1:1 API details (journal, OA, venue, tldr) are flattened onto s2_paper.
-- Content columns are mostly nullable: citation/reference stubs are sparse.

-- s2_paper — one row per Semantic Scholar paperId (subject or related).
create table public.s2_paper (
    paper_id                     text primary key,
    corpus_id                    bigint,
    title                        text,
    abstract                     text,
    venue                        text,
    year                         integer,
    url                          text,
    reference_count              integer,
    citation_count               integer,
    influential_citation_count   integer,
    is_open_access               boolean,
    publication_date             text,
    publication_types            text[],
    fields_of_study              text[],
    external_ids                 jsonb,
    citation_styles              jsonb,
    -- flattened journal
    journal_name                 text,
    journal_volume               text,
    journal_pages                text,
    -- flattened openAccessPdf
    oa_url                       text,
    oa_status                    text,
    oa_license                   text,
    oa_disclaimer                text,
    -- flattened publicationVenue
    venue_id                     text,
    venue_name                   text,
    venue_type                   text,
    venue_issn                   text,
    venue_url                    text,
    venue_alternate_names        text[],
    venue_alternate_urls         text[],
    -- flattened tldr
    tldr_model                   text,
    tldr_text                    text,
    -- [{category, source}, ...]
    s2_fields_of_study           jsonb,
    fetched_at                   timestamptz
);
alter table public.s2_paper enable row level security;

comment on table public.s2_paper is
    'Semantic Scholar paper nodes (subject + related). Natural key = S2 paperId.';

-- s2_author — one row per S2 authorId.
create table public.s2_author (
    author_id       text primary key,
    name            text,
    url             text,
    affiliations    text[],
    homepage        text,
    paper_count     integer,
    citation_count  integer,
    h_index         integer,
    external_ids    jsonb
);
alter table public.s2_author enable row level security;

comment on table public.s2_author is
    'Semantic Scholar author nodes. Natural key = S2 authorId.';

-- s2_paper_author — authorship with order (0-based position).
create table public.s2_paper_author (
    paper_id   text not null references public.s2_paper (paper_id) on delete cascade,
    author_id  text not null references public.s2_author (author_id) on delete cascade,
    position   integer not null,
    primary key (paper_id, author_id),
    check (position >= 0)
);
create index s2_paper_author_author_idx on public.s2_paper_author (author_id);
alter table public.s2_paper_author enable row level security;

comment on table public.s2_paper_author is
    'Authorship: paper ↔ author with 0-based author order from the API list.';

-- s2_reference — outgoing: paper_id cites referenced_paper_id.
create table public.s2_reference (
    paper_id             text not null references public.s2_paper (paper_id) on delete cascade,
    referenced_paper_id  text not null references public.s2_paper (paper_id) on delete cascade,
    primary key (paper_id, referenced_paper_id),
    check (paper_id <> referenced_paper_id)
);
create index s2_reference_referenced_idx on public.s2_reference (referenced_paper_id);
alter table public.s2_reference enable row level security;

comment on table public.s2_reference is
    'Outgoing cites: paper_id cites referenced_paper_id. Query refs of P with where paper_id = P.';

-- s2_citation — incoming: citing_paper_id cites paper_id (stored under cited paper).
create table public.s2_citation (
    paper_id          text not null references public.s2_paper (paper_id) on delete cascade,
    citing_paper_id   text not null references public.s2_paper (paper_id) on delete cascade,
    primary key (paper_id, citing_paper_id),
    check (paper_id <> citing_paper_id)
);
create index s2_citation_citing_idx on public.s2_citation (citing_paper_id);
alter table public.s2_citation enable row level security;

comment on table public.s2_citation is
    'Incoming cites: citing_paper_id cites paper_id. Query citations of P with where paper_id = P. '
    'Isolates highly-cited fan-in from s2_reference.';

-- s2_embedding — optional SPECTER vector; side table to avoid fat paper rows.
-- specter_v1 is 768-d; pgvector enables cosine / L2 KNN (index when query volume warrants it).
create extension if not exists vector with schema extensions;

create table public.s2_embedding (
    paper_id    text primary key references public.s2_paper (paper_id) on delete cascade,
    model       text not null,
    embedding   extensions.vector(768) not null,
    fetched_at  timestamptz not null default now()
);
alter table public.s2_embedding enable row level security;

comment on table public.s2_embedding is
    'Optional S2 paper embedding (specter_v1 = vector(768)). Not fetched by default harvest fields.';
