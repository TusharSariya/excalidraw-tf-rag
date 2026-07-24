-- source_openreview (type key 'openreview'). Per-seed OUTCOME row for the
-- OpenReview harvest source, mirroring source_arxiv / source_openalex: created
-- pending when the source is dispatched for a seed, flipped to succeeded with the
-- typed fields filled in, or not_found / error. Every data column is nullable —
-- populated only on success. paper_id is a bigint FK to seeds; on delete cascade.
--
-- Typed for the fields the OpenReview Note projection reads
-- (id, title, abstract, authors, pdf, venueid); raw jsonb keeps the whole note.
create table public.source_openreview (
    paper_id       bigint primary key references public.seeds (paper_id) on delete cascade,
    status         source_status not null default 'pending',
    openreview_id  text,
    title          text,
    abstract       text,
    authors        text[] not null default '{}',
    pdf            text,
    venueid        text,
    forum          text,
    number         integer,
    raw            jsonb,
    error          text,          -- populated when status = 'error'
    fetched_at     timestamptz    -- set when status = 'succeeded'
);
alter table public.source_openreview enable row level security;

-- Register the source: one queue + one subscription row. The filter gates
-- fan-out to seeds that carry an openreview_id (validated by
-- harvest_filter_is_valid, which already allows 'openreview_id').
select pgmq.create('harvest_openreview');
insert into public.queue_subscriptions (queue_name, source, filter) values
    ('harvest_openreview', 'openreview', '{"any_of": ["openreview_id"]}'::jsonb);
