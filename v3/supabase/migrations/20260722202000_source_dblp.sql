-- source_dblp (type key 'dblp') — per-seed OUTCOME of the dblp record fetch.
-- Mirrors DblpRecord in v3/harvest/src/harvest/sources/models/dblp.py. Typed for
-- the fields we query now; the full XML response is kept verbatim on raw
-- ({"xml": <text>} — dblp has no JSON variant). Every data column is nullable:
-- it is populated only on success. paper_id is a bigint FK to seeds; on delete
-- cascade drops a seed's outcome with the seed. dblp needs no API key.
create table public.source_dblp (
    paper_id     bigint primary key references public.seeds (paper_id) on delete cascade,
    status       source_status not null default 'pending',
    dblp_id      text,
    record_type  text,          -- 'inproceedings' (conf) | 'article' (journal/CoRR) | ...
    title        text,
    authors      jsonb,         -- [{name, orcid}] ordered
    venue        text,          -- <booktitle> (conf) or <journal> (article)
    year         integer,
    ee           text[],        -- electronic-edition URLs
    doi          text,          -- derived from an ee on doi.org; frequently null
    raw          jsonb,         -- {"xml": <full response>}
    error        text,          -- populated when status = 'error'
    fetched_at   timestamptz    -- set when status = 'succeeded'
);
alter table public.source_dblp enable row level security;

-- One queue per source: creates pgmq.q_harvest_dblp (live) + pgmq.a_harvest_dblp (archive).
select pgmq.create('harvest_dblp');

-- Subscribe the source. dblp resolves ONLY a dblp record key — its filter gates
-- dispatch to seeds that carry dblp_id (validated by harvest_filter_is_valid,
-- which allows dblp_id since 20260722100000_seed_identity_columns).
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_dblp', 'dblp', '{"any_of": ["dblp_id"]}'::jsonb);
