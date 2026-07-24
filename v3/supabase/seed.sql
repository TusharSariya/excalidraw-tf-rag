-- seed.sql — LOCAL / CI prod-parity setup (NOT applied to prod).
--
-- Runs automatically after every `supabase db reset` / `supabase start` via
-- config.toml [db.seed] sql_paths. It is NEVER applied to the hosted project:
-- prod deploys migrations with `supabase db push`, which does not run seed.sql.
-- That is exactly why the pgmq_public wrappers live here and not in a migration —
-- on prod they are installed out of band by the dashboard "Queues" integration,
-- and a migration creating them would collide with the dashboard-managed ones.
--
-- Three things a bare local/CI stack lacks that the hosted project provides, all
-- needed for harvest-work + the seeder to run locally (and for PostgREST to come
-- up at all: config.toml exposes `pgmq_public` in the Data API `schemas`, so the
-- schema MUST exist by the time PostgREST loads its cache or it 503s):
--
--   1. the pgmq_public.* wrappers (read / send / archive / pop) — on PROD the
--      dashboard "Queues" integration installs them; a local stack has none.
--   2. DML privileges for the API `service_role` on the public harvest tables —
--      the REST seeder writes public.seeds (and reads it back) as service_role.
--      On the hosted platform service_role is auto-granted DML on public tables;
--      a local stack's default privileges grant the API roles only
--      TRUNCATE/REFERENCES/TRIGGER, so `insert into public.seeds` fails with
--      "permission denied for table seeds" (42501).
--   3. pgmq schema access for `service_role` — inserting a seed fires the
--      seeds_identity_inserted trigger, whose public.publish_seed (SECURITY
--      INVOKER) calls pgmq.send directly. pgmq's own functions are SECURITY
--      INVOKER too, so the fan-out runs as service_role and needs USAGE on
--      schema pgmq + INSERT on the pgmq.q_* live tables; without it the seed
--      insert fails with "permission denied for schema pgmq" (42501).
--
-- The bodies mirror the canonical Supabase Queues wrappers verbatim: each one
-- runs with `set search_path = ''` and delegates to the underlying pgmq.*
-- function with fully-qualified names. Returning `setof pgmq.message_record`
-- (rather than an inlined column list) keeps this correct across pgmq point
-- releases — 1.5 added a `headers` column to that composite type, and by naming
-- the type instead of its columns we inherit whichever shape the locally
-- installed pgmq ships.

create schema if not exists pgmq_public;

-- read(queue_name, sleep_seconds, n): make up to n messages invisible for
-- sleep_seconds and return them. pgmq.read's params are (queue_name, vt, qty);
-- the wrapper renames vt->sleep_seconds and qty->n to match the worker's call
-- in harvest/persistence/supabase.py (read_batch).
create or replace function pgmq_public.read(
    queue_name text,
    sleep_seconds integer,
    n integer
)
returns setof pgmq.message_record
language plpgsql
set search_path = ''
as $$
begin
    return query
    select *
    from pgmq.read(
        queue_name := queue_name,
        vt := sleep_seconds,
        qty := n
    );
end;
$$;

-- send(queue_name, message, sleep_seconds=0): enqueue one message, optionally
-- delayed by sleep_seconds; returns the new msg_id. pgmq.send's params are
-- (queue_name, msg, delay); the wrapper renames msg->message and delay->
-- sleep_seconds. (The seeders publish via public.publish_seed -> pgmq.send
-- server-side and do not call this wrapper directly, but the dashboard
-- integration always installs it, so this mirrors prod.)
create or replace function pgmq_public.send(
    queue_name text,
    message jsonb,
    sleep_seconds integer default 0
)
returns setof bigint
language plpgsql
set search_path = ''
as $$
begin
    return query
    select *
    from pgmq.send(
        queue_name := queue_name,
        msg := message,
        delay := sleep_seconds
    );
end;
$$;

-- archive(queue_name, message_id): move one message from the live queue to its
-- archive table; returns whether a row was archived. This is the worker's ack
-- (harvest/worker.py _archive). pgmq.archive's params are (queue_name, msg_id);
-- the wrapper renames msg_id->message_id.
create or replace function pgmq_public.archive(
    queue_name text,
    message_id bigint
)
returns boolean
language plpgsql
set search_path = ''
as $$
begin
    return pgmq.archive(
        queue_name := queue_name,
        msg_id := message_id
    );
end;
$$;

-- pop(queue_name): atomically read-and-delete the next visible message.
-- Included for parity with the prod dashboard wrapper set; delegates straight
-- to pgmq.pop(queue_name).
create or replace function pgmq_public.pop(queue_name text)
returns setof pgmq.message_record
language plpgsql
set search_path = ''
as $$
begin
    return query
    select *
    from pgmq.pop(queue_name := queue_name);
end;
$$;

-- Expose the wrappers to the Data API roles, matching the prod grants.
grant usage on schema pgmq_public to service_role, anon, authenticated;

grant execute on function pgmq_public.read(text, integer, integer)
    to service_role, anon, authenticated;
grant execute on function pgmq_public.send(text, jsonb, integer)
    to service_role, anon, authenticated;
grant execute on function pgmq_public.archive(text, bigint)
    to service_role, anon, authenticated;
grant execute on function pgmq_public.pop(text)
    to service_role, anon, authenticated;

-- --------------------------------------------------------------------------
-- service_role DML on the public harvest tables (see header note 2).
--
-- Granted on ALL current tables in schema public rather than an enumerated list:
-- every source adds a new source_<key> table, and an enumerated grant would be
-- the same per-source maintenance trap the capability-gated fan-out exists to
-- avoid. Because seed.sql runs AFTER `supabase db reset` has applied every
-- migration, "all tables" already covers seeds, queue_subscriptions, and every
-- source_* table that exists. Sequences are granted too so any non-identity
-- serial/default the seeder touches resolves. service_role only — these tables
-- are backend-only, never anon/authenticated facing (they carry no RLS policy).
grant select, insert, update, delete
    on all tables in schema public to service_role;
grant usage, select
    on all sequences in schema public to service_role;

-- --------------------------------------------------------------------------
-- service_role access to schema pgmq (see header note 3). publish_seed's
-- pgmq.send runs as the invoking service_role, so it needs schema USAGE, EXECUTE
-- on the pgmq functions, and DML on the pgmq.q_* live tables + their msg-id
-- sequences. The q_* tables already exist here: every source migration's
-- pgmq.create(...) ran during `supabase db reset`, before seed.sql.
grant usage on schema pgmq to service_role;
grant execute on all functions in schema pgmq to service_role;
grant select, insert, update, delete on all tables in schema pgmq to service_role;
grant usage, select on all sequences in schema pgmq to service_role;
