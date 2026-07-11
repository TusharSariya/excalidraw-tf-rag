-- One table per harvest source. Each holds that source's per-seed OUTCOME:
-- a row is created (status='pending') when the source is attempted for a seed,
-- flips to 'succeeded' with the typed data filled in, or 'not_found'/'error'.
-- This is why every data column is nullable — it's populated only on success.
-- paper_id is a bigint FK to seeds; because seed identity is per-corpus, these
-- rows are per-corpus too. on delete cascade: dropping a seed drops its outcomes.

create type public.source_status as enum ('pending', 'succeeded', 'not_found', 'error');

-- source_arxiv — mirrors ArxivPaper in v3/harvest/src/harvest/sources/arxiv.py
create table public.source_arxiv (
    paper_id          bigint primary key references public.seeds (paper_id) on delete cascade,
    status            source_status not null default 'pending',
    entry_id          text,
    title             text,
    authors           text[] not null default '{}',
    summary           text,
    primary_category  text,
    categories        text[] not null default '{}',
    pdf_url           text,
    links             text[] not null default '{}',
    published         timestamptz,
    updated           timestamptz,
    comment           text,
    journal_ref       text,
    doi               text,
    error             text,          -- populated when status = 'error'
    fetched_at        timestamptz    -- set when status = 'succeeded'
);
alter table public.source_arxiv enable row level security;

-- source_openalex — typed for the fields we use now; raw jsonb keeps the rest.
create table public.source_openalex (
    paper_id          bigint primary key references public.seeds (paper_id) on delete cascade,
    status            source_status not null default 'pending',
    openalex_id       text,
    doi               text,
    title             text,
    publication_date  date,
    type              text,
    best_oa_pdf_url   text,
    cited_by_count    integer,
    authorships       jsonb,
    raw               jsonb,
    error             text,
    fetched_at        timestamptz
);
alter table public.source_openalex enable row level security;

-- source_semantic_scholar (type key 's2'). corpus_id here is S2's own corpus id,
-- unrelated to seeds.corpus.
create table public.source_semantic_scholar (
    paper_id             bigint primary key references public.seeds (paper_id) on delete cascade,
    status               source_status not null default 'pending',
    s2_paper_id          text,
    corpus_id            bigint,
    title                text,
    external_ids         jsonb,
    open_access_pdf_url  text,
    reference_count      integer,
    citation_count       integer,
    fields_of_study      text[],
    raw                  jsonb,
    error                text,
    fetched_at           timestamptz
);
alter table public.source_semantic_scholar enable row level security;
