-- source_figshare — per-seed OUTCOME row for the figshare harvest source.
-- Mirrors source_openalex: keyed by the seed's bigint paper_id, typed for the
-- fields we query now, raw jsonb keeps the whole article. Every data column is
-- nullable — populated only on success. on delete cascade: dropping a seed drops
-- its outcome.
--
-- Identity note: `resource_doi` is the seed's publication DOI (the field figshare
-- search matches on); `doi` is figshare's OWN minted deposit DOI (e.g. …g001),
-- which is a DIFFERENT identifier and must not be confused with the seed's.
create table public.source_figshare (
    paper_id           bigint primary key references public.seeds (paper_id) on delete cascade,
    status             source_status not null default 'pending',
    figshare_id        bigint,
    doi                text,
    resource_doi       text,
    title              text,
    resource_title     text,
    description        text,
    defined_type_name  text,
    figshare_url       text,
    url_public_html    text,
    published_date     timestamptz,
    version            integer,
    download_url       text,
    license_name       text,
    tags               text[],
    files              jsonb,
    authors            jsonb,
    raw                jsonb,
    error              text,
    fetched_at         timestamptz
);
alter table public.source_figshare enable row level security;

-- Queue + subscription: figshare resolves from a publication DOI (seed doi).
select pgmq.create('harvest_figshare');
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_figshare', 'figshare', '{"any_of": ["doi"]}'::jsonb);
