-- Make the KEDA scaler able to SEE the queue. Fixes a silent, total failure.
--
-- The problem: Supabase enables RLS on the pgmq queue tables (pgmq.q_<queue>) and
-- ships NO policies on them — access is meant to go through the SECURITY DEFINER
-- `pgmq_public` wrappers. harvester_keda_ro is a plain role without BYPASSRLS, so
-- RLS filters every row away regardless of the GRANT SELECT in
-- 20260711050000_harvester_keda_ro_role.sql. GRANTs do not override RLS.
--
-- pgmq.metrics() is SECURITY INVOKER, so called as harvester_keda_ro it counted an
-- empty table and returned queue_length = 0. Always. No error, no warning — the
-- scaler reported "queue empty" no matter how deep the real backlog was, and KEDA
-- sat at zero replicas reporting ScalerNotActive. A silent wrong answer, which is
-- the worst kind: an unreachable database at least logs something.
--
-- The fix mirrors what Supabase itself does for pgmq: a narrow SECURITY DEFINER
-- wrapper. Rejected alternatives:
--   * `alter role harvester_keda_ro bypassrls` — needs superuser, and would let the
--     scaler read past RLS on EVERY table in the database. A least-privilege role
--     should not hold the most dangerous flag in Postgres to count messages.
--   * RLS policies on each pgmq.q_* table — pgmq.create() makes a new table per
--     queue, so every future queue silently re-breaks until someone remembers.
--
-- Verified against the live staging project on 2026-07-16.

create or replace function public.harvest_queue_visible_length(p_queue text)
returns bigint
language plpgsql
security definer
-- Pin search_path: a SECURITY DEFINER function without it can be hijacked by a
-- caller-controlled search_path resolving these names to their own objects.
set search_path = pg_catalog, pgmq, public
as $$
declare
    n bigint;
begin
    -- Only queues we actually publish to. Without this the function is an arbitrary
    -- read oracle over any queue in the database for whoever can execute it.
    if not exists (
        select 1 from public.queue_subscriptions
        where queue_name = p_queue and active
    ) then
        raise exception 'unknown or inactive queue: %', p_queue;
    end if;

    select queue_visible_length into n from pgmq.metrics(p_queue);
    return coalesce(n, 0);
end $$;

-- Default-deny, then grant only the scaler. `public` includes anon/authenticated in
-- a Supabase project, so the revoke is load-bearing, not decorative.
revoke all on function public.harvest_queue_visible_length(text) from public;
grant execute on function public.harvest_queue_visible_length(text) to harvester_keda_ro;

comment on function public.harvest_queue_visible_length(text) is
    'Visible message count for an active harvest queue. SECURITY DEFINER: the KEDA '
    'scaler role cannot read pgmq.q_* directly because Supabase enables RLS there '
    'with no policies. Execute is granted to harvester_keda_ro only.';
