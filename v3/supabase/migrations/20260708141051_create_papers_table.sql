create type public.harvest_status as enum ('pending', 'succeeded', 'exhausted');

create table public.papers (
    paper_id       uuid primary key default gen_random_uuid(),
    openalex_id    text unique,
    doi            text unique,
    arxiv_id       text unique,
    title_hash     text,
    status         harvest_status not null default 'pending',
    storage_path   text,
    backup_path    text,
    first_seen_at  timestamptz not null default now(),
    harvested_at   timestamptz
);

alter table public.papers enable row level security;
-- no policies added: locked down to service-role/backend access only
