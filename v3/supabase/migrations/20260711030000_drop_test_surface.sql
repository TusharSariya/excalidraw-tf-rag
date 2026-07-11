-- Drop the _test mirror surface created by 20260711020343_create_test_surface.
--
-- The mirror was a hand-maintained copy of the harvest catalog + fan-out surface
-- so an e2e test could run against shared staging without touching real data. It
-- has been superseded: the fan-out e2e now runs against the REAL objects on a
-- disposable ephemeral database (the local `supabase start` stack, per PR in CI),
-- so no in-database mirror is needed. Remove it to end the schema duplication.
--
-- Forward-only: the create migration stays in history (already applied on
-- staging); this drops the objects it created. Order: function, then tables
-- (cascade clears the source_*_test FKs), then the pgmq queues.

drop function if exists public.publish_seed_test(bigint);

drop table if exists
    public.source_arxiv_test,
    public.source_openalex_test,
    public.source_semantic_scholar_test,
    public.queue_subscriptions_test,
    public.seeds_test
    cascade;

select pgmq.drop_queue('harvest_arxiv_test');
select pgmq.drop_queue('harvest_openalex_test');
select pgmq.drop_queue('harvest_s2_test');
