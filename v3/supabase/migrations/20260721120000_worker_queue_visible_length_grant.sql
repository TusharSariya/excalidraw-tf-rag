-- Let the harvest worker peek queue depth for batch sizing.
--
-- The worker connects to PostgREST with the Supabase service_role key, so its SQL
-- runs as the `service_role` Postgres role. It needs the visible message count to
-- decide whether a full batch is already available (read + flush in one shot) vs.
-- falling back to the accumulate-with-timer path when the stream is trickling.
--
-- The count itself can only come from pgmq, and pgmq.metrics() is unreachable over
-- PostgREST (pgmq is not an exposed schema; only public and pgmq_public are). So the
-- worker calls a public-schema wrapper via rpc() instead.
--
-- public.harvest_queue_visible_length() was first added for the KEDA scaler in
-- 20260716190000. It is (re)defined here with `create or replace` so this migration
-- is self-contained; the body is identical, so applying both in order is a no-op the
-- second time. Keep the two in sync if either changes — it is one canonical object.
--
-- SECURITY DEFINER: the raw pgmq.q_* tables have RLS enabled with no policies, so a
-- non-BYPASSRLS caller (the KEDA scaler role) counts an empty table. Running as the
-- owner sidesteps that. The queue_subscriptions gate stops it being a read-oracle
-- over arbitrary queues.

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

-- Default-deny, then grant the two callers: the KEDA scaler (unchanged from
-- 20260716190000) and the harvest worker (service_role, added here).
revoke all on function public.harvest_queue_visible_length(text) from public;
grant execute on function public.harvest_queue_visible_length(text) to harvester_keda_ro;
grant execute on function public.harvest_queue_visible_length(text) to service_role;

comment on function public.harvest_queue_visible_length(text) is
    'Visible message count for an active harvest queue. SECURITY DEFINER: callers '
    'cannot read pgmq.q_* directly because Supabase enables RLS there with no '
    'policies. Execute granted to harvester_keda_ro (scaler) and service_role (worker).';
