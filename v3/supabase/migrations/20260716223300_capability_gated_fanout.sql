-- Capability-gated fan-out. Uses the tables from 20260716223236_harvest_dispatch.
--
-- publish_seed used to enqueue to EVERY active queue unconditionally. An arXiv-only
-- seed therefore landed in harvest_openalex, where the worker cannot succeed —
-- OpenAlex has no arXiv filter, and arXiv's own DOI (10.48550/arxiv.<id>) does not
-- resolve there either (both lookup forms verified 404 on 2026-07-16). The worker then
-- wrote status='not_found', which is a LIE: the paper is not missing from OpenAlex; we
-- never held an identity to ask with. Those two states demand different responses and
-- the old schema could not distinguish them.
--
-- Now: a source is dispatched only when the seed carries identity that source can
-- actually use (its `filter`), and only once per generation (harvest_dispatch).
-- Full control flow: docs/architecture/harvest-dispatch.md

-- Which generation is this source on? coalesce so that adding a subscription later
-- does not require remembering to seed harvest_generation.
create or replace function public.harvest_current_generation(p_source text)
returns int
language sql
stable
as $$
    select coalesce(
        (select generation from public.harvest_generation where source = p_source), 1
    )
$$;

-- A typo'd field name in a filter must fail LOUDLY, at configuration time. Evaluated
-- lazily it would simply match nothing, and the source would silently never run —
-- indistinguishable from "no seed has that identity yet". That silent-no-op class of
-- bug is the entire reason this migration exists, so it is not permitted to reappear
-- via a misspelled filter.
create or replace function public.harvest_filter_is_valid(p_filter jsonb)
returns boolean
language sql
immutable
as $$
    select p_filter is null              -- null = unconditional (back-compat)
        or (
            jsonb_typeof(p_filter -> 'any_of') = 'array'
            and jsonb_array_length(p_filter -> 'any_of') > 0
            and (
                select bool_and(x in ('arxiv_id', 'doi', 'openalex_id', 'title'))
                from jsonb_array_elements_text(p_filter -> 'any_of') as t(x)
            )
        )
$$;

alter table public.queue_subscriptions
    add constraint queue_subscriptions_filter_valid
    check (public.harvest_filter_is_valid(filter));

-- Does this seed carry identity satisfying the filter?
-- "any_of" = the source can work from ANY ONE of these (S2 takes an arXiv id OR a doi).
create or replace function public.seed_satisfies(s public.seeds, p_filter jsonb)
returns boolean
language sql
stable
as $$
    select p_filter is null
        or exists (
            select 1
            from jsonb_array_elements_text(p_filter -> 'any_of') as t(field)
            where (field = 'arxiv_id'    and s.arxiv_id    is not null)
               or (field = 'doi'         and s.doi         is not null)
               or (field = 'openalex_id' and s.openalex_id is not null)
               or (field = 'title'       and s.title       is not null)
        )
$$;

-- Win the right to enqueue exactly one message for (paper_id, source).
--
-- Race safety is the ON CONFLICT row lock, not application logic: two concurrent
-- publish_seed calls for the same pair serialise here. The loser's DO UPDATE ... WHERE
-- evaluates (generation < generation) = false, so no row is updated, RETURNING yields
-- nothing, `claimed` stays NULL -> false. Exactly one message is sent.
--
-- Do NOT rewrite this as SELECT-then-INSERT; that reintroduces the race it closes.
create or replace function public.harvest_claim_dispatch(p_paper_id bigint, p_source text)
returns boolean
language plpgsql
as $$
declare
    claimed boolean;
begin
    insert into public.harvest_dispatch (paper_id, source, generation)
    values (p_paper_id, p_source, public.harvest_current_generation(p_source))
    on conflict (paper_id, source) do update
        set generation    = excluded.generation,
            dispatched_at = now(),
            queue_msg_id  = null            -- superseded; the new send sets it
        where public.harvest_dispatch.generation < excluded.generation
    returning true into claimed;

    return coalesce(claimed, false);
end $$;

-- Fan a seed out to every source that can use its CURRENT identity and has not already
-- run at this generation. The seeder still never names a queue.
--
-- The envelope stays thin ({v, paper_id}) on purpose: the worker re-reads seeds at
-- execution time, so a message enqueued when we held only an arXiv id will see a DOI
-- that landed while it sat in the queue. A fat message would be a stale snapshot.
create or replace function public.publish_seed(p_paper_id bigint) returns void
language plpgsql as $$
declare
    s   public.seeds%rowtype;
    q   record;
    msg jsonb := jsonb_build_object('v', 1, 'paper_id', p_paper_id);
    mid bigint;
begin
    if not jsonb_matches_schema(public.seed_message_schema(), msg) then
        raise exception 'seed message failed v1 schema validation: %', msg;
    end if;

    select * into s from public.seeds where paper_id = p_paper_id;
    if not found then
        raise exception 'publish_seed: no seed with paper_id %', p_paper_id;
    end if;

    for q in select queue_name, source, filter from public.queue_subscriptions where active loop
        if public.seed_satisfies(s, q.filter) then          -- gate 1: can it run?
            if public.harvest_claim_dispatch(p_paper_id, q.source) then   -- gate 2: has it?
                mid := pgmq.send(q.queue_name, msg);
                update public.harvest_dispatch
                   set queue_msg_id = mid
                 where paper_id = p_paper_id and source = q.source;
            end if;
        end if;
    end loop;
end $$;

-- Identity growth is the event. A worker that learns a DOI writes it to seeds; this
-- re-runs the fan-out so newly-satisfied sources dispatch. The cascade is therefore a
-- property of the DATA, not of worker discipline — a worker that forgets to re-publish
-- cannot break it.
create or replace function public.publish_seed_trigger() returns trigger
language plpgsql as $$
begin
    perform public.publish_seed(new.paper_id);
    return null;                      -- AFTER trigger; return value ignored
end $$;

-- Two triggers, not one `insert or update`: a WHEN clause on an INSERT cannot reference
-- OLD (it does not exist), and TG_OP is a PL/pgSQL variable available only INSIDE the
-- function — not in WHEN. A combined trigger fails to create with
-- `column "tg_op" does not exist`.

-- Every new seed publishes: there is no OLD to compare, and identity is by definition new.
create trigger seeds_identity_inserted
after insert on public.seeds
for each row
execute function public.publish_seed_trigger();

-- WHEN matters here: without it, ANY update re-runs the fan-out — including a worker
-- rewriting the same DOI after a pgmq redelivery. With it, a no-op update is silent, so
-- redelivery costs nothing and cannot re-arm the cascade.
create trigger seeds_identity_changed
after update on public.seeds
for each row
when (
    old.doi         is distinct from new.doi
    or old.arxiv_id    is distinct from new.arxiv_id
    or old.openalex_id is distinct from new.openalex_id
    or old.title       is distinct from new.title
)
execute function public.publish_seed_trigger();

-- Identity is MONOTONIC: fields fill NULL -> value and never change or clear. This is
-- what bounds the cascade — each seed can fire at most once per identity field, so it
-- provably terminates rather than probably terminating.
--
-- It also implements "OpenAlex must never clobber a good DOI" in one line. We measured
-- OpenAlex returning a mirror's DOI for 2 of 3 papers; one such overwrite would both
-- poison every DOI-keyed source downstream AND re-arm the cascade.
create or replace function public.seeds_identity_monotonic() returns trigger
language plpgsql as $$
begin
    new.doi         := coalesce(old.doi, new.doi);
    new.arxiv_id    := coalesce(old.arxiv_id, new.arxiv_id);
    new.openalex_id := coalesce(old.openalex_id, new.openalex_id);
    new.title       := coalesce(old.title, new.title);
    return new;
end $$;

create trigger seeds_identity_monotonic
before update on public.seeds
for each row
execute function public.seeds_identity_monotonic();

-- Declare what each source needs. Adding a source stays an INSERT here — no code change.
--
--   arxiv    : the arXiv API is keyed by arXiv id. Nothing else.
--   s2       : accepts ARXIV:<id> or DOI:<doi>. Most reachable source we have.
--   openalex : DOI or OpenAlex id ONLY. Deliberately NOT title — title search returned
--              the wrong paper for 2 of 5 sampled arXiv ids (0/4 and 0/10 author
--              overlap), and a wrong record persists as confidently as a right one.
update public.queue_subscriptions set filter = '{"any_of": ["arxiv_id"]}'::jsonb
 where source = 'arxiv';
update public.queue_subscriptions set filter = '{"any_of": ["arxiv_id", "doi"]}'::jsonb
 where source = 's2';
update public.queue_subscriptions set filter = '{"any_of": ["doi", "openalex_id"]}'::jsonb
 where source = 'openalex';

-- Re-ingest, scoped to one source: bump the generation, then sweep.
--
-- The sweep is NOT optional. The trigger fires on seeds changes, and bumping a
-- generation does not touch seeds — so nothing would re-dispatch without it. Wrapped
-- here so that is impossible to forget.
create or replace function public.harvest_republish(p_source text, p_reason text default null)
returns int
language plpgsql as $$
declare
    n int;
begin
    insert into public.harvest_generation (source, generation, reason)
    values (p_source, 2, p_reason)
    on conflict (source) do update
        set generation = public.harvest_generation.generation + 1,
            bumped_at  = now(),
            reason     = coalesce(excluded.reason, public.harvest_generation.reason);

    -- Only seeds that can actually run this source get re-dispatched; publish_seed's
    -- filter gate still applies, so a title-only seed stays untouched.
    perform public.publish_seed(paper_id) from public.seeds;

    select count(*) into n
      from public.harvest_dispatch
     where source = p_source
       and generation = public.harvest_current_generation(p_source);
    return n;
end $$;
