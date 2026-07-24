-- Purge a Supabase project's harvest state: seeds, everything downstream, and the
-- harvest queue messages — so nothing is left pointing at a seed that no longer exists.
--
-- Clears, in one transaction (RESTART IDENTITY so seeds.paper_id starts over):
--   * seeds
--   * every source_* outcome table (FK-cascade from seeds)
--   * harvest_dispatch (FK-cascade from seeds)
--   * crossref_reference / opencitations_edge (edge fan-outs)
--   * s2_* graph tables (not seed-linked)
--   * every message in the harvest_* pgmq queues (pgmq.purge_queue)
--
-- Preserves the queue / fan-out control-plane — only the mailboxes are emptied:
--   * the pgmq queues themselves
--   * queue_subscriptions (fan-out config)
--   * harvest_generation (per-source dispatch epoch)
-- ...and any migration / schema state.
--
-- CASCADE is a safety net for any FK child not named below; every table we
-- actually expect to clear is listed explicitly so this reads as an inventory.
-- Keep this list in sync with the `tables` array in purge_staging.sh.

begin;

truncate table
    public.seeds,
    public.source_arxiv,
    public.source_openalex,
    public.source_semantic_scholar,
    public.source_unpaywall,
    public.source_crossref,
    public.source_dblp,
    public.source_openreview,
    public.source_acl_anthology,
    public.source_opencitations,
    public.source_datacite,
    public.source_core,
    public.source_internet_archive,
    public.source_zenodo,
    public.source_figshare,
    public.source_osf,
    public.crossref_reference,
    public.opencitations_edge,
    public.harvest_dispatch,
    public.s2_paper,
    public.s2_author,
    public.s2_paper_author,
    public.s2_reference,
    public.s2_citation,
    public.s2_embedding
restart identity cascade;

-- Empty the harvest queue mailboxes (messages only). purge_queue keeps the queue,
-- its subscription, and harvest_generation — it just deletes pending / orphaned
-- messages that would otherwise point at now-truncated seeds.
select pgmq.purge_queue(queue_name)
from pgmq.list_queues()
where queue_name like 'harvest\_%';

commit;
