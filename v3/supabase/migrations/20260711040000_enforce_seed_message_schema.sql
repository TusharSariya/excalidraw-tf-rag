-- Codify the seed -> message contract in the database and enforce it at the
-- single write path (public.publish_seed), so a malformed or drifted payload
-- raises instead of being shipped to consumers.
--
-- The canonical schema also lives, byte-for-byte, at
--   v3/harvest/src/harvest/contracts/seed_message.v1.json
-- which the Python harvest clients load. seed_message_schema() below is the
-- in-database copy (SQL migrations can't read files); a test asserts the two
-- stay in sync. Keep them identical when either changes.
--
-- Forward-only: replaces the publish_seed body from
-- 20260711004017_replace_papers_with_seeds. The message now carries a versioned
-- envelope {"v": 1, "paper_id": <bigint>} instead of a bare {"paper_id": ...}.

create extension if not exists pg_jsonschema;

-- The v1 seed-message JSON Schema, as the database sees it. Mirrors
-- contracts/seed_message.v1.json exactly.
create or replace function public.seed_message_schema() returns json
language sql
immutable
parallel safe
as $$
    select $schema$
    {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "$id": "https://excalidraw-tf-rag/harvest/seed_message/v1",
      "title": "Harvest seed message (v1)",
      "description": "The envelope public.publish_seed enqueues onto every active harvest_* queue. Thin: it carries only the seed's bigint paper_id; a consumer resolves identity (arxiv_id/doi/corpus) from public.seeds. The `v` field versions the wire format so the payload can evolve without breaking existing consumers.",
      "type": "object",
      "required": ["v", "paper_id"],
      "additionalProperties": false,
      "properties": {
        "v": {
          "const": 1,
          "description": "Envelope schema version."
        },
        "paper_id": {
          "type": "integer",
          "minimum": 1,
          "description": "bigint identity from public.seeds.paper_id."
        }
      }
    }
    $schema$::json;
$$;

-- Fan one seed out to every active queue, stamping + validating the v1 envelope
-- before each send. The seeder still calls only this; it never names a queue.
create or replace function public.publish_seed(p_paper_id bigint) returns void
language plpgsql as $$
declare
    q   record;
    msg jsonb := jsonb_build_object('v', 1, 'paper_id', p_paper_id);
begin
    if not jsonb_matches_schema(public.seed_message_schema(), msg) then
        raise exception 'seed message failed v1 schema validation: %', msg;
    end if;
    for q in select queue_name from public.queue_subscriptions where active loop
        perform pgmq.send(q.queue_name, msg);
    end loop;
end $$;
