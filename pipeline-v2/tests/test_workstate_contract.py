"""Contract tests for WorkStateRepository, run against both backends.

The ``repo`` fixture (tests/conftest.py) is parametrized over
InMemoryWorkStateRepository (always) and PostgresWorkStateRepository
(``@pytest.mark.integration``, needs docker + RUN_INTEGRATION=1) so every
test in this file exercises the real invariant on both backends without
duplicating the test body.
"""

from __future__ import annotations

import pytest
from pipeline_v2.workstate import WorkStateError


def _outcome(
    repo, doc_id="doc-1", corpus="glr", stage="embed", sha="sha-a", fp="fp-a", state="ok", **kwargs
):
    return repo.upsert_outcome(
        doc_id=doc_id,
        corpus=corpus,
        stage=stage,
        content_sha256=sha,
        stage_fingerprint=fp,
        state=state,
        **kwargs,
    )


def test_check_constraint_rejects_invalid_state(repo):
    with pytest.raises(WorkStateError):
        repo.upsert_outcome(
            doc_id="doc-1",
            corpus="glr",
            stage="embed",
            content_sha256="sha-a",
            stage_fingerprint="fp-a",
            state="not-a-real-state",  # type: ignore[arg-type]
        )


def test_degraded_row_is_not_indexable(repo):
    _outcome(repo, state="degraded", reason="ocr fallback used")

    assert repo.list_indexable(corpus="glr") == []
    dead = repo.get_dead_letters(corpus="glr")
    assert len(dead) == 1
    assert dead[0].state == "degraded"


def test_ok_row_is_indexable(repo):
    result = _outcome(repo, state="ok")
    assert result.changed is True

    indexable = repo.list_indexable(corpus="glr")
    assert len(indexable) == 1
    assert indexable[0].doc_id == "doc-1"
    assert indexable[0].state == "ok"


def test_idempotent_rerun_same_sha_fingerprint_ok_is_noop(repo):
    first = _outcome(repo, state="ok")
    assert first.changed is True
    assert first.row.attempt == 1

    second = _outcome(repo, state="ok")
    assert second.changed is False
    assert second.row.attempt == 1  # unchanged, no new attempt recorded

    third = repo.get("doc-1", "glr", "embed")
    assert third is not None
    assert third.attempt == 1


def test_rerun_with_changed_fingerprint_is_a_new_attempt(repo):
    _outcome(repo, state="ok", fp="fp-a")
    second = _outcome(repo, state="ok", fp="fp-b")

    assert second.changed is True
    assert second.row.attempt == 2
    assert second.row.stage_fingerprint == "fp-b"


def test_failed_then_ok_records_new_attempt(repo):
    _outcome(repo, state="failed", reason="timeout")
    ok = _outcome(repo, state="ok")

    assert ok.changed is True
    assert ok.row.attempt == 2
    assert ok.row.state == "ok"
    assert ok.row.reason is None


def test_retry_dead_letter_touches_only_non_ok(repo):
    repo.upsert_outcome(
        doc_id="doc-ok",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="ok",
    )
    repo.upsert_outcome(
        doc_id="doc-failed",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="failed",
        reason="boom",
    )
    repo.upsert_outcome(
        doc_id="doc-degraded",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="degraded",
        reason="fallback",
    )
    # Different stage: must not be touched by a retry scoped to "embed".
    repo.upsert_outcome(
        doc_id="doc-failed",
        corpus="glr",
        stage="extract",
        content_sha256="s",
        stage_fingerprint="f",
        state="failed",
        reason="boom-extract",
    )

    touched = repo.retry_dead_letter("embed", corpus="glr")
    assert touched == 2

    ok_row = repo.get("doc-ok", "glr", "embed")
    assert ok_row.attempt == 1  # untouched

    failed_row = repo.get("doc-failed", "glr", "embed")
    assert failed_row.attempt == 2
    assert failed_row.reason is None
    assert failed_row.state == "failed"  # still not indexable

    degraded_row = repo.get("doc-degraded", "glr", "embed")
    assert degraded_row.attempt == 2
    assert degraded_row.reason is None
    assert degraded_row.state == "degraded"

    other_stage_row = repo.get("doc-failed", "glr", "extract")
    assert other_stage_row.attempt == 1  # different stage, untouched
    assert other_stage_row.reason == "boom-extract"

    # Retried rows are still dead letters, still not indexable.
    indexable_ids = {row.doc_id for row in repo.list_indexable(corpus="glr")}
    assert indexable_ids == {"doc-ok"}


def test_get_dead_letters_filters_by_stage_and_corpus(repo):
    repo.upsert_outcome(
        doc_id="a",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="failed",
    )
    repo.upsert_outcome(
        doc_id="b",
        corpus="rlr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="failed",
    )
    repo.upsert_outcome(
        doc_id="c",
        corpus="glr",
        stage="extract",
        content_sha256="s",
        stage_fingerprint="f",
        state="degraded",
    )

    assert {r.doc_id for r in repo.get_dead_letters(stage="embed")} == {"a", "b"}
    assert {r.doc_id for r in repo.get_dead_letters(stage="embed", corpus="glr")} == {"a"}
    assert {r.doc_id for r in repo.get_dead_letters(corpus="glr")} == {"a", "c"}


def test_claim_retry_returns_dead_letters_only(repo):
    repo.upsert_outcome(
        doc_id="ok-doc",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="ok",
    )
    repo.upsert_outcome(
        doc_id="failed-doc",
        corpus="glr",
        stage="embed",
        content_sha256="s",
        stage_fingerprint="f",
        state="failed",
    )

    with repo.claim_retry("embed", corpus="glr", limit=10) as claimed:
        claimed_ids = {row.doc_id for row in claimed}

    assert claimed_ids == {"failed-doc"}
