# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
# ]
# ///
"""End-to-end proof of capability-gated fan-out and the identity cascade.

Runs against the REAL objects (`public.seeds`, `public.publish_seed`, and the
`harvest_arxiv` / `harvest_openalex` / `harvest_s2` queues) over a single direct
Postgres connection — the way the real seeder actually works (it connects
directly and calls `publish_seed`; nothing in the fan-out path goes through
PostgREST). One connection does the write, the verify, and the purge.

What it proves (see docs/architecture/harvest-dispatch.md):

  Phase 1 — a seed carrying ONLY an arXiv id fans out to exactly the sources
  that can use it: harvest_arxiv and harvest_s2. harvest_openalex gets NOTHING,
  because OpenAlex cannot be queried by arXiv id (its filter is {doi,
  openalex_id}). The absent message is the CORRECT answer, not a failure — an
  arXiv-only seed landing in harvest_openalex is the exact bug the
  capability-gated design removed.

  Phase 2 — when a DOI is written to the seed (as the s2 worker would after
  resolving one), the identity-changed trigger re-runs publish_seed. Now
  harvest_openalex is satisfied and receives the paper; harvest_arxiv and
  harvest_s2 do NOT get a second message, because harvest_claim_dispatch already
  recorded them at this generation. The cascade is a property of the data.

DESTRUCTIVE: this purges the real harvest queues and truncates `public.seeds`.
It is meant for a DISPOSABLE, ephemeral database only — the local `supabase
start` stack, or a throwaway preview branch — NEVER staging or prod. As a
guard it refuses to run unless `SUPABASE_DB_URL` points at a loopback host;
set `E2E_ALLOW_NONLOCAL=1` to override for a deliberately-disposable remote DB.

Run it:

    SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \\
    uv run v3/supabase/tests/e2e_fanout.py

Exits 0 when both phases match the expected per-queue delivery, then everything
is purged; non-zero on any mismatch.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

import psycopg

CORPUS = "e2e-test"
ARXIV_ID = "1706.03762"
# Synthetic published DOI written in phase 2 to trigger the cascade. Its only
# job is to be non-null so the OpenAlex filter ({doi, openalex_id}) is satisfied;
# publish_seed never dereferences it.
DOI = "10.18653/v1/e2e-test-doi"

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


def snapshot(cur: psycopg.Cursor) -> dict[str, int | None]:
    """Map each queue to the paper_id of its single message, or None if empty.

    A queue holding anything other than exactly one message maps to a sentinel
    string so the comparison against the expected dict fails loudly rather than
    silently collapsing two messages to one paper_id.
    """
    result: dict[str, int | None] = {}
    for queue_name, q_table in QUEUES:
        cur.execute(f"select message from {q_table}")  # noqa: S608 (fixed identifiers)
        rows = cur.fetchall()
        if len(rows) == 0:
            result[queue_name] = None
        elif len(rows) == 1:
            result[queue_name] = rows[0][0].get("paper_id")
        else:
            result[queue_name] = f"<{len(rows)} messages>"  # type: ignore[assignment]
    return result


def check(label: str, got: dict[str, int | None], expected: dict[str, int | None]) -> list[str]:
    """Print the snapshot and return a list of human-readable mismatches."""
    print(f"\n{label}")
    print("queue                  got        expected")
    print("---------------------  ---------  ---------")
    failures: list[str] = []
    for queue_name, _ in QUEUES:
        g = got[queue_name]
        e = expected[queue_name]
        mark = "" if g == e else "   <-- MISMATCH"
        print(f"{queue_name:<21}  {str(g):<9}  {str(e):<9}{mark}")
        if g != e:
            failures.append(f"{label}: {queue_name}: expected {e!r}, got {g!r}")
    return failures


def main() -> int:
    db_url = _require_env("SUPABASE_DB_URL")
    _guard_disposable(db_url)

    failures: list[str] = []

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

            # 3. Phase 1 — arXiv id only. Capability gate blocks OpenAlex.
            failures += check(
                "PHASE 1 — arXiv-only seed (OpenAlex must be empty):",
                snapshot(cur),
                {"harvest_arxiv": paper_id, "harvest_openalex": None, "harvest_s2": paper_id},
            )

            # 4. A DOI arrives (as the s2 worker would write it). The
            #    identity-changed trigger re-runs the fan-out.
            cur.execute("update public.seeds set doi = %s where paper_id = %s", (DOI, paper_id))
            print(f"\nupdated seeds: paper_id={paper_id} doi={DOI} (cascade should fire)")

            # 5. Phase 2 — OpenAlex now satisfied; arxiv/s2 not re-sent (claim).
            failures += check(
                "PHASE 2 — DOI added (cascade reaches OpenAlex, others unchanged):",
                snapshot(cur),
                {"harvest_arxiv": paper_id, "harvest_openalex": paper_id, "harvest_s2": paper_id},
            )

        # 6. Finale purge — leave the DB empty regardless of outcome.
        print("\nfinale: purging harvest queues + seeds")
        purge(conn)

    if failures:
        print("\nFAIL — fan-out did not match the capability-gated design:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nOK — capability gate + cascade both verified for paper_id={paper_id}, then purged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
