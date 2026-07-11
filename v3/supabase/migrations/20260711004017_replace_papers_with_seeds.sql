-- Switch the catalog identity from papers(uuid) to seeds(bigint identity).
-- papers is empty and has no dependents yet (no source_* / harvest_attempts
-- tables reference it), so drop and replace rather than migrate data.
drop table if exists public.papers cascade;

-- The catalog: one row per known paper. paper_id is a monotonically increasing
-- identity; every external identifier is optional (a submission may carry any
-- one of arxiv/doi/openalex, or only a title). The unique constraints make each
-- external id a natural dedup key for on-conflict upserts.
create table public.seeds (
    paper_id       bigint generated always as identity primary key,
    openalex_id    text unique,
    doi            text unique,
    arxiv_id       text unique,
    title_hash     text,
    title          text,
    -- operational state (carried from papers; the worker writes these)
    status         harvest_status not null default 'pending',
    storage_path   text,
    backup_path    text,
    first_seen_at  timestamptz not null default now(),
    harvested_at   timestamptz
);

alter table public.seeds enable row level security;
-- no policies: locked down to service-role/backend access only

-- publish_seed is now keyed by the bigint identity. Replace the uuid version.
drop function if exists public.publish_seed(uuid);
create function public.publish_seed(p_paper_id bigint) returns void
language plpgsql as $$
declare q record;
begin
    for q in select queue_name from public.queue_subscriptions where active loop
        perform pgmq.send(q.queue_name, jsonb_build_object('paper_id', p_paper_id));
    end loop;
end $$;
