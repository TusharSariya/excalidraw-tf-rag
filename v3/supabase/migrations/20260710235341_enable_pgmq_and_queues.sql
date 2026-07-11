create extension if not exists pgmq;

-- One queue per source. Each creates pgmq.q_<name> (live) + pgmq.a_<name> (archive).
select pgmq.create('harvest_arxiv');
select pgmq.create('harvest_openalex');
select pgmq.create('harvest_s2');

-- SNS-style subscription registry: adding a source = INSERT a row, no code change.
create table public.queue_subscriptions (
    queue_name text primary key,
    source     text not null,          -- type key: 'arxiv' | 'openalex' | 's2'
    active     boolean not null default true,
    filter     jsonb                   -- optional SNS-style filter policy (unused for now)
);
insert into public.queue_subscriptions (queue_name, source) values
    ('harvest_arxiv',    'arxiv'),
    ('harvest_openalex', 'openalex'),
    ('harvest_s2',       's2');

alter table public.queue_subscriptions enable row level security;

-- The "topic": fan one paper out to every active queue, atomically, in one transaction.
-- The seeder calls only this; it never names a queue.
create function public.publish_seed(p_paper_id uuid) returns void
language plpgsql as $$
declare q record;
begin
    for q in select queue_name from public.queue_subscriptions where active loop
        perform pgmq.send(q.queue_name, jsonb_build_object('paper_id', p_paper_id::text));
    end loop;
end $$;
