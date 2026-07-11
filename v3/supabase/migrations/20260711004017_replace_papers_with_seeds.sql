-- Switch the catalog identity from papers(uuid) to seeds(bigint identity).
-- papers is empty and has no dependents yet, so drop and replace rather than
-- migrate data.
drop table if exists public.papers cascade;

-- The catalog is deliberately DUMB: identity only, no lifecycle state. A seed is
-- what we push into the queues; per-source status/data lives in the source_*
-- tables (one per source), because a paper has an independent outcome per source.
--
-- paper_id is a monotonically increasing identity. Every external identifier is
-- optional (a submission may carry any one of arxiv/doi/openalex, or only a
-- title). corpus is required: the same paper can belong to more than one corpus,
-- so identity is per-corpus — dedup keys are (corpus, <id>), and the same paper
-- harvested for two corpora is two rows with two paper_ids. NULL ids don't
-- collide (Postgres treats NULLs as distinct), so a title-only seed is fine.
create table public.seeds (
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

alter table public.seeds enable row level security;
-- no policies: locked down to service-role/backend access only

-- publish_seed is now keyed by the bigint identity. Replace the uuid version.
drop function if exists public.publish_seed(uuid);
create or replace function public.publish_seed(p_paper_id bigint) returns void
language plpgsql as $$
declare q record;
begin
    for q in select queue_name from public.queue_subscriptions where active loop
        perform pgmq.send(q.queue_name, jsonb_build_object('paper_id', p_paper_id));
    end loop;
end $$;
