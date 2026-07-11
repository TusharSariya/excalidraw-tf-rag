-- A _test-suffixed MIRROR of the real harvest catalog + fan-out surface. Staging
-- pulls double duty: the real objects serve actual staging, this _test surface
-- exists purely so an end-to-end test can seed, publish, verify the fan-out, and
-- purge everything WITHOUT touching real data or the real harvest_* queues.
--
-- It mirrors the shape of the real objects exactly (see 20260711004017 and
-- 20260711011845) so the test exercises the same structure prod runs. The only
-- reuse from the real schema is the public.source_status enum — no _test copy.

-- seeds_test — mirror of public.seeds (dumb identity-only catalog, per-corpus).
create table public.seeds_test (
    paper_id     bigint generated always as identity primary key,
    corpus       text not null,
    openalex_id  text,
    doi          text,
    arxiv_id     text,
    title_hash   text,
    title        text,
    unique (corpus, openalex_id),
    unique (corpus, doi),
    unique (corpus, arxiv_id)
);
alter table public.seeds_test enable row level security;

-- Per-source outcome mirrors. paper_id FKs to seeds_test; cascade on delete so
-- truncating seeds_test in the test teardown clears these too.
create table public.source_arxiv_test (
    paper_id          bigint primary key references public.seeds_test (paper_id) on delete cascade,
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
    error             text,
    fetched_at        timestamptz
);
alter table public.source_arxiv_test enable row level security;

create table public.source_openalex_test (
    paper_id          bigint primary key references public.seeds_test (paper_id) on delete cascade,
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
alter table public.source_openalex_test enable row level security;

create table public.source_semantic_scholar_test (
    paper_id             bigint primary key references public.seeds_test (paper_id) on delete cascade,
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
alter table public.source_semantic_scholar_test enable row level security;

-- Test queues — separate from the real harvest_* queues so the test can purge
-- them freely without ever touching real pending work.
select pgmq.create('harvest_arxiv_test');
select pgmq.create('harvest_openalex_test');
select pgmq.create('harvest_s2_test');

-- Test subscription registry — mirror of public.queue_subscriptions.
create table public.queue_subscriptions_test (
    queue_name text primary key,
    source     text not null,
    active     boolean not null default true,
    filter     jsonb
);
insert into public.queue_subscriptions_test (queue_name, source) values
    ('harvest_arxiv_test',    'arxiv'),
    ('harvest_openalex_test', 'openalex'),
    ('harvest_s2_test',       's2');
alter table public.queue_subscriptions_test enable row level security;

-- publish_seed_test — the test "topic": fan one paper out to every active test
-- queue, in one transaction. Mirror of public.publish_seed, over the _test
-- registry. Staging now carries exactly two publish functions.
--
-- SECURITY DEFINER (owner = the migration role, which has pgmq access): a caller
-- that only holds the API `service_role` has EXECUTE on pgmq.send but NOT USAGE
-- on schema pgmq nor INSERT on the pgmq.q_* tables, so an INVOKER function would
-- fail to fan out when called over REST. Running as the owner lets the
-- service-role seeder trigger the fan-out. search_path is pinned empty and every
-- object fully qualified (the standard SECURITY DEFINER hardening).
-- NOTE: the real public.publish_seed is SECURITY INVOKER and has the same gap —
-- it can't currently be fanned out by the service_role. Flagged separately.
create or replace function public.publish_seed_test(p_paper_id bigint) returns void
language plpgsql
security definer
set search_path = ''
as $$
declare q record;
begin
    for q in select queue_name from public.queue_subscriptions_test where active loop
        perform pgmq.send(q.queue_name, jsonb_build_object('paper_id', p_paper_id));
    end loop;
end $$;

-- The write path is the Supabase client with the `service_role` key. On cloud,
-- Supabase's default privileges auto-grant DML on new public tables to the API
-- roles; the local stack does NOT, so grant explicitly here to keep the test
-- portable (identical behaviour local, staging, and CI). Only service_role — the
-- _test surface is backend-only, never anon/authenticated facing.
grant select, insert, update, delete on
    public.seeds_test,
    public.source_arxiv_test,
    public.source_openalex_test,
    public.source_semantic_scholar_test
    to service_role;
