-- Least-privilege role for the KEDA postgresql scaler.
--
-- KEDA scales the harvest workers by polling queue depth:
--   SELECT queue_visible_length FROM pgmq.metrics('<queue>')
-- over the Supabase session pooler (:5432). That scaler connects as this role —
-- it needs to read pgmq metrics and nothing else (the worker itself never uses
-- this role; it talks to the queues over the REST pgmq_public wrappers).
--
-- Created with LOGIN but NO password here — deliberately. Set the real password
-- out-of-band (Supabase SQL editor, not committed) before wiring up the scaler:
--   alter role harvester_keda_ro password '<generated>';

create role harvester_keda_ro login;

grant usage on schema pgmq to harvester_keda_ro;
grant execute on function pgmq.metrics(text) to harvester_keda_ro;
grant select on all tables in schema pgmq to harvester_keda_ro;

-- Cover queues added after this migration too (new pgmq.q_*/a_* tables).
alter default privileges in schema pgmq grant select on tables to harvester_keda_ro;
