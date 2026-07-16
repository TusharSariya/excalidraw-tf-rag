-- Fix an incomplete grant in 20260711050000_harvester_keda_ro_role.sql.
--
-- That migration granted SELECT on all TABLES in pgmq, which is necessary but not
-- sufficient: pgmq.metrics() also reads each queue's message-id sequence
-- (pgmq.q_<queue>_msg_id_seq) to compute total_messages. In Postgres, sequences are
-- a separate privilege class — `GRANT SELECT ON ALL TABLES` does not cover them.
--
-- So the role could execute metrics() and read the queue tables, but the function
-- failed partway through with:
--
--   permission denied for sequence q_harvest_arxiv_msg_id_seq
--
-- KEDA surfaces that as a scaler error and holds every ScaledJob at zero replicas —
-- which is indistinguishable from an empty queue unless you read the operator log.
-- Verified against the live staging project on 2026-07-16.

grant select on all sequences in schema pgmq to harvester_keda_ro;

-- Cover queues added after this migration too (each pgmq.create() makes a new
-- q_<queue>_msg_id_seq), mirroring the default privileges already set for tables.
alter default privileges in schema pgmq grant select on sequences to harvester_keda_ro;
