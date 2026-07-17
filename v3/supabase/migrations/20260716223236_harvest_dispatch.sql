create table public.harvest_generation (
    source text primary key,
    generation int not null default 1 check (generation > 0),
    bumped_at timestamptz not null default now(),
    reason text
);
alter table public.harvest_generation enable row level security;

insert into public.harvest_generation (source, reason)
select source, 'initial' from public.queue_subscriptions;

create table public.harvest_dispatch (
    paper_id      bigint not null references public.seeds (paper_id) on delete cascade,
    source        text   not null,
    generation    int    not null,
    dispatched_at timestamptz not null default now(),
    queue_msg_id  bigint,
    primary key (paper_id, source)
);
alter table public.harvest_dispatch enable row level security;

create index harvest_dispatch_source_gen_idx
    on public.harvest_dispatch (source, generation);
