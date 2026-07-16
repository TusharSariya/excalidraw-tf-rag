# /// script
# requires-python = ">=3.11"
# dependencies = ["supabase>=2.15"]
# ///
"""Submit one seed and fan it out to every active harvest queue.

This is the stopgap for `harvest-seed`, which the harvest README documents but
which does not exist yet (no entrypoint in pyproject.toml, no module — only the
unused `SeederSettings` class). It does what that entrypoint is specified to do:
insert into `public.seeds`, then call `public.publish_seed(paper_id)`, which stamps
and pg_jsonschema-validates the v1 envelope before enqueuing to each active queue.

NON-DESTRUCTIVE, unlike `../tests/e2e_fanout.py` — that one purges the queues and
truncates `public.seeds`, and is for a disposable local stack only. This one only
ever inserts, so it is safe against staging.

Reaches the DB over PostgREST with the service key, because `public.seeds` has RLS
enabled with no policies (service-role/backend only). The real seeder is specified
to connect directly to Postgres instead; this is equivalent for the fan-out, since
publish_seed does all the work server-side either way.

Run it:

    export SUPABASE_URL=https://<ref>.supabase.co
    export SUPABASE_SERVICE_KEY=$(supabase projects api-keys \
        --project-ref <ref> --reveal -o json \
        | python3 -c 'import json,sys; print(next(k["api_key"] for k in json.load(sys.stdin) if k["name"]=="v3harvest"))')

    uv run v3/supabase/scripts/seed.py --corpus drain-test --arxiv-id 1706.03762

Prints the new paper_id and the queues it was published to. It does not report queue
depth — see the comment at the end of main() for why that would corrupt read_ct.
"""

from __future__ import annotations

import argparse
import os
import sys

from supabase import create_client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="corpus this seed belongs to (required)")
    ap.add_argument("--arxiv-id", help="arxiv identifier, e.g. 1706.03762")
    ap.add_argument("--doi")
    ap.add_argument("--openalex-id")
    ap.add_argument("--title")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        print(
            "set SUPABASE_URL and SUPABASE_SERVICE_KEY (see this file's docstring)", file=sys.stderr
        )
        return 2

    ids = {
        "arxiv_id": args.arxiv_id,
        "doi": args.doi,
        "openalex_id": args.openalex_id,
        "title": args.title,
    }
    ids = {k: v for k, v in ids.items() if v}
    if not ids:
        print("give at least one of --arxiv-id / --doi / --openalex-id / --title", file=sys.stderr)
        return 2

    c = create_client(url, key)

    # seeds is UNIQUE per (corpus, <id>), so re-seeding the same paper into the same
    # corpus conflicts. Reuse the existing row rather than failing: the point is to
    # re-enqueue, and identity is per-corpus by design.
    existing = c.table("seeds").select("paper_id").eq("corpus", args.corpus)
    for k, v in ids.items():
        if k != "title":
            existing = existing.eq(k, v)
    rows = existing.execute().data if len(ids) > 1 or "title" not in ids else []

    if rows:
        paper_id = rows[0]["paper_id"]
        print(
            f"seed already exists in corpus {args.corpus!r} -> paper_id={paper_id} (re-publishing)"
        )
    else:
        inserted = c.table("seeds").insert({"corpus": args.corpus, **ids}).execute().data
        paper_id = inserted[0]["paper_id"]
        print(f"inserted seed -> paper_id={paper_id}")

    c.schema("public").rpc("publish_seed", {"p_paper_id": paper_id}).execute()

    subs = c.table("queue_subscriptions").select("queue_name").eq("active", True).execute().data
    print(
        f"published paper_id={paper_id} to {len(subs)} active queue(s): "
        f"{', '.join(s['queue_name'] for s in subs)}"
    )

    # Deliberately NOT printing queue depths here. The only depth call available to
    # this key is pgmq_public.read (pgmq_public exposes no metrics wrapper), and read
    # increments read_ct even with sleep_seconds=0 — so peeking would inflate the
    # redelivery counter that `read_ct` exists to report, corrupting the signal you
    # would use to spot a poison message looping. A diagnostic must not mutate what
    # it measures. To check depth, use public.harvest_queue_visible_length('<queue>')
    # in the SQL editor, or just watch the workers:
    #   kubectl logs -n harvester -l 'scaledjob.keda.sh/name' --prefix --tail=-1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
