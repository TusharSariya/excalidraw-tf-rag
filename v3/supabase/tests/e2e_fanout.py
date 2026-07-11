# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
# ]
# ///
"""End-to-end proof that a seed fans out to every harvest queue.

Runs against the REAL objects (`public.seeds`, `public.publish_seed`, and the
`harvest_arxiv` / `harvest_openalex` / `harvest_s2` queues) over a single direct
Postgres connection — the way the real seeder actually works (it connects
directly as `harvester_worker` and calls the SECURITY INVOKER `publish_seed`;
nothing in the fan-out path goes through PostgREST). One connection does the
write, the verify, and the purge.

DESTRUCTIVE: this purges the real harvest queues and truncates `public.seeds`.
It is meant for a DISPOSABLE, ephemeral database only — the local `supabase
start` stack, or a throwaway preview branch — NEVER staging or prod. As a
guard it refuses to run unless `SUPABASE_DB_URL` points at a loopback host;
set `E2E_ALLOW_NONLOCAL=1` to override for a deliberately-disposable remote DB.

Run it:

    SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \\
    uv run v3/supabase/tests/e2e_fanout.py

Exits 0 on success (all three queues received exactly the published paper_id,
then everything is purged), non-zero on any mismatch.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

import psycopg

CORPUS = "e2e-test"
ARXIV_ID = "1706.03762"

# (subscription queue name, backing pgmq live table). publish_seed sends to the
# queue names on the left; pgmq stores them in pgmq.q_<name> on the right.
QUEUES = [
    ("harvest_arxiv", "pgmq.q_harvest_arxiv"),
    ("harvest_openalex", "pgmq.q_harvest_openalex"),
    ("harvest_s2", "pgmq.q_harvest_s2"),
]

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: ${name} is required")
    return value


def _guard_disposable(db_url: str) -> None:
    """Refuse to touch a non-loopback DB unless explicitly overridden.

    This test purges real queues and truncates real seeds; running it against
    staging/prod would wipe live data. Only a disposable ephemeral DB is safe.
    """
    host = (urlsplit(db_url).hostname or "").lower()
    if host in LOOPBACK_HOSTS:
        return
    if os.environ.get("E2E_ALLOW_NONLOCAL") == "1":
        print(f"warning: E2E_ALLOW_NONLOCAL=1 — running destructively against host {host!r}")
        return
    sys.exit(
        f"refusing to run: SUPABASE_DB_URL host {host!r} is not loopback. This test is "
        "destructive (purges real queues, truncates seeds) and is only safe on a "
        "disposable ephemeral DB. Set E2E_ALLOW_NONLOCAL=1 to override for a throwaway DB."
    )


def purge(conn: psycopg.Connection) -> None:
    """Empty the harvest queues and the seed catalog. Idempotent."""
    with conn.cursor() as cur:
        for queue_name, _ in QUEUES:
            cur.execute("select pgmq.purge_queue(%s)", (queue_name,))
        # cascade clears the source_* rows too
        cur.execute("truncate public.seeds cascade")


def main() -> int:
    db_url = _require_env("SUPABASE_DB_URL")
    _guard_disposable(db_url)

    with psycopg.connect(db_url, autocommit=True) as conn:
        # 1. Pre-clean so the post-publish counts are exact.
        print("pre-clean: purging harvest queues + seeds")
        purge(conn)

        with conn.cursor() as cur:
            # 2. Write + publish the same way the seeder does: direct pg.
            cur.execute(
                "insert into public.seeds (corpus, arxiv_id) values (%s, %s) returning paper_id",
                (CORPUS, ARXIV_ID),
            )
            paper_id = cur.fetchone()[0]
            print(f"seeded seeds: paper_id={paper_id} corpus={CORPUS} arxiv_id={ARXIV_ID}")

            cur.execute("select public.publish_seed(%s)", (paper_id,))
            print(f"published paper_id={paper_id} via publish_seed")

            # 3. Verify each queue got exactly the published paper_id.
            failures: list[str] = []
            print("\nqueue                  depth  paper_id")
            print("---------------------  -----  --------")
            for queue_name, q_table in QUEUES:
                cur.execute(f"select message from {q_table}")  # noqa: S608 (fixed identifiers)
                rows = cur.fetchall()
                depth = len(rows)
                got = rows[0][0].get("paper_id") if depth == 1 else None
                ok = depth == 1 and got == paper_id
                print(f"{queue_name:<21}  {depth:>5}  {got if got is not None else '-'}")
                if not ok:
                    failures.append(
                        f"{queue_name}: expected depth 1 with paper_id={paper_id}, "
                        f"got depth {depth} paper_id={got}"
                    )

        # 4. Finale purge — leave the DB empty regardless of outcome.
        print("\nfinale: purging harvest queues + seeds")
        purge(conn)

    if failures:
        print("\nFAIL — fan-out did not reach every queue:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nOK — paper_id={paper_id} fanned out to all {len(QUEUES)} queues, then purged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
