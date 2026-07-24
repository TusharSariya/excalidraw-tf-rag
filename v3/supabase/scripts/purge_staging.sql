-- Purge a Supabase project's harvest state: seeds, everything downstream, and the
-- harvest queue messages — so nothing is left pointing at a seed that no longer exists.
--
-- Clears, in one transaction (RESTART IDENTITY so seeds.paper_id starts over):
--   * seeds
--   * source_arxiv / source_openalex / source_semantic_scholar  (FK-cascade from seeds)
--   * harvest_dispatch                                          (FK-cascade from seeds)
--   * s2_paper / s2_author / s2_paper_author / s2_reference /
--     s2_citation / s2_embedding                                (graph; not seed-linked)
--   * every message in the harvest_* pgmq queues                (pgmq.purge_queue)
--
-- Preserves the queue / fan-out control-plane — only the mailboxes are emptied:
--   * the pgmq queues themselves (harvest_arxiv / harvest_openalex / harvest_s2)
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
