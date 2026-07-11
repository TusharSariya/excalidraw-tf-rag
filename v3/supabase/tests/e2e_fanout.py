# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "supabase>=2.31.0",
#   "psycopg[binary]>=3.2",
# ]
# ///
"""End-to-end proof that a seed fans out to every harvest queue.

Runs against the `_test` mirror surface (see the create_test_surface migration)
so it never touches real data or the real harvest_* queues. Faithful to how the
real system works: the WRITE goes through the Supabase client (the way the
seeder pushes), while VERIFY + PURGE go over a direct Postgres connection (the
way the KEDA scaler / worker actually read pgmq — nothing in prod touches pgmq
via REST).

Run it:

    SUPABASE_URL=...  SUPABASE_KEY=<service_role key>  SUPABASE_DB_URL=postgresql://... \\
    uv run v3/supabase/tests/e2e_fanout.py

Exits 0 on success (all three test queues received exactly the published
paper_id, then everything is purged), non-zero on any mismatch.
"""

from __future__ import annotations

import os
import sys

import psycopg
from supabase import create_client

CORPUS = "e2e-test"
ARXIV_ID = "1706.03762"

# (subscription queue name, backing pgmq live table). publish_seed_test sends to
# the queue names on the left; pgmq stores them in pgmq.q_<name> on the right.
TEST_QUEUES = [
    ("harvest_arxiv_test", "pgmq.q_harvest_arxiv_test"),
    ("harvest_openalex_test", "pgmq.q_harvest_openalex_test"),
    ("harvest_s2_test", "pgmq.q_harvest_s2_test"),
]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: ${name} is required")
    return value


def purge(conn: psycopg.Connection) -> None:
    """Empty the test queues and the test catalog. Idempotent."""
    with conn.cursor() as cur:
        for queue_name, _ in TEST_QUEUES:
            cur.execute("select pgmq.purge_queue(%s)", (queue_name,))
        # cascade clears the source_*_test rows too
        cur.execute("truncate public.seeds_test cascade")


def main() -> int:
    supabase_url = _require_env("SUPABASE_URL")
    supabase_key = _require_env("SUPABASE_KEY")
    db_url = _require_env("SUPABASE_DB_URL")

    with psycopg.connect(db_url, autocommit=True) as conn:
        # 1. Pre-clean so the post-publish counts are exact.
        print("pre-clean: purging test queues + seeds_test")
        purge(conn)

        # 2. Write via the Supabase client (service key bypasses RLS).
        sb = create_client(supabase_url, supabase_key)
        inserted = sb.table("seeds_test").insert({"corpus": CORPUS, "arxiv_id": ARXIV_ID}).execute()
        paper_id = inserted.data[0]["paper_id"]
        print(f"seeded seeds_test: paper_id={paper_id} corpus={CORPUS} arxiv_id={ARXIV_ID}")

        sb.rpc("publish_seed_test", {"p_paper_id": paper_id}).execute()
        print(f"published paper_id={paper_id} via publish_seed_test")

        # 3. Verify each test queue got exactly the published paper_id.
        failures: list[str] = []
        print("\nqueue                       depth  paper_id")
        print("--------------------------  -----  --------")
        with conn.cursor() as cur:
            for queue_name, q_table in TEST_QUEUES:
                cur.execute(f"select message from {q_table}")  # noqa: S608 (fixed identifiers)
                rows = cur.fetchall()
                depth = len(rows)
                got = rows[0][0].get("paper_id") if depth == 1 else None
                ok = depth == 1 and got == paper_id
                print(f"{queue_name:<26}  {depth:>5}  {got if got is not None else '-'}")
                if not ok:
                    failures.append(
                        f"{queue_name}: expected depth 1 with paper_id={paper_id}, "
                        f"got depth {depth} paper_id={got}"
                    )

        # 4. Finale purge — leave the _test surface empty regardless of outcome.
        print("\nfinale: purging test queues + seeds_test")
        purge(conn)

    if failures:
        print("\nFAIL — fan-out did not reach every queue:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"\nOK — paper_id={paper_id} fanned out to all {len(TEST_QUEUES)} test queues, then purged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
