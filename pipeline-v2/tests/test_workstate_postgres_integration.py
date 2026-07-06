"""Postgres-only integration tests that need real cross-connection concurrency.

Contract tests shared with the in-memory backend live in
test_workstate_contract.py (parametrized over the `repo` fixture). Real
`SELECT ... FOR UPDATE SKIP LOCKED` concurrency can only be demonstrated
against an actual database with two separate connections/transactions, so
those live here instead.
"""

from __future__ import annotations

import pytest
from pipeline_v2.workstate import PostgresWorkStateRepository

pytestmark = pytest.mark.integration


def test_claim_retry_skip_locked_excludes_concurrent_claim(clean_postgres_dsn):
    repo_a = PostgresWorkStateRepository(clean_postgres_dsn)
    repo_a.init_db()

    for doc_id in ("doc-1", "doc-2"):
        repo_a.upsert_outcome(
            doc_id=doc_id,
            corpus="glr",
            stage="embed",
            content_sha256="s",
            stage_fingerprint="f",
            state="failed",
        )

    repo_b = PostgresWorkStateRepository(clean_postgres_dsn)

    # repo_a claims and holds both dead-letter rows without releasing the
    # locks yet (manual __enter__/__exit__ instead of `with`, so we can run
    # repo_b's claim while repo_a's transaction is still open).
    claim_a = repo_a.claim_retry("embed", corpus="glr", limit=10)
    claimed_a = claim_a.__enter__()
    assert {row.doc_id for row in claimed_a} == {"doc-1", "doc-2"}

    try:
        # repo_b, on a separate connection, must skip everything repo_a is
        # still holding — SKIP LOCKED means it sees zero claimable rows.
        with repo_b.claim_retry("embed", corpus="glr", limit=10) as claimed_b:
            assert claimed_b == []
    finally:
        claim_a.__exit__(None, None, None)  # releases repo_a's row locks

    # Once repo_a released its locks, the rows are claimable again.
    with repo_b.claim_retry("embed", corpus="glr", limit=10) as claimed_b_after:
        assert {row.doc_id for row in claimed_b_after} == {"doc-1", "doc-2"}


def test_check_constraint_is_enforced_at_the_database_level(clean_postgres_dsn):
    """Belt-and-suspenders: even bypassing the Python-level guard, the DB CHECK holds."""
    import psycopg

    repo = PostgresWorkStateRepository(clean_postgres_dsn)
    repo.init_db()

    with psycopg.connect(clean_postgres_dsn) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO work_state
                    (doc_id, corpus, content_sha256, stage, stage_fingerprint, state)
                VALUES ('d', 'glr', 's', 'embed', 'f', 'pending')
                """
            )
