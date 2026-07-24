-- source_internet_archive (type key 'internet_archive'). Per-seed OUTCOME of the
-- Internet Archive metadata fetch, keyed by the seed's bigint paper_id. As with every
-- source table, data columns are nullable — populated only on success (status flips
-- 'pending' -> 'succeeded'/'not_found'/'error'). Mirrors InternetArchiveItem in
-- v3/harvest/src/harvest/sources/models/internet_archive.py.
--
-- IA has no citation graph and no native batch endpoint: one item per metadata request,
-- one row per seed. pdf_url is the CONSTRUCTED archive.org/download URL — the worker
-- persists it but fetches metadata only, never the binary.
create table public.source_internet_archive (
    paper_id       bigint primary key references public.seeds (paper_id) on delete cascade,
    status         source_status not null default 'pending',
    ia_identifier  text,
    title          text,
    creator        text[] not null default '{}',
    date           text,
    publicdate     text,
    year           integer,
    description    text,
    subject        text[] not null default '{}',
    licenseurl     text,
    mediatype      text,
    pdf_file_name  text,
    pdf_url        text,
    raw            jsonb,
    error          text,          -- populated when status = 'error'
    fetched_at     timestamptz    -- set when status = 'succeeded'
);
alter table public.source_internet_archive enable row level security;

-- Queue + subscription: adding a source is an INSERT, no code change. The filter gates
-- fan-out so only seeds carrying an ia_identifier dispatch here (see
-- 20260716223300_capability_gated_fanout.sql for why the gate exists).
select pgmq.create('harvest_internet_archive');
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_internet_archive', 'internet_archive', '{"any_of": ["ia_identifier"]}'::jsonb);
