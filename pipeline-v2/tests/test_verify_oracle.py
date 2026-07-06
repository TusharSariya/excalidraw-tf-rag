"""Tests for the `pipeline-v2 verify` invariant oracle (rearchitecture T0d).

The non-negotiable here is the must-fail self-test: the oracle has to prove
it actually catches a violation, not just rubber-stamp whatever it's handed.
``test_degraded_row_indexed_is_caught`` reproduces the exact shape of the v1
bug that destroyed 2,266 docs (a non-'ok' work_state outcome, but the doc
present in the index anyway) and asserts the DATA layer reports FAIL.

Uses the ``repo`` fixture from conftest.py (parametrized memory/postgres;
postgres param is marked integration and only runs under RUN_INTEGRATION=1)
plus a real Lance table written through the production write path
(``make_index_writer`` / ``_index_one``), so these tests exercise the same
on-disk shape the indexer stage itself produces.
"""

from __future__ import annotations

import json

import lancedb
import pytest
from pipeline_v2 import verify
from pipeline_v2.artifacts import LocalArtifactStore
from pipeline_v2.contracts import StageMessage
from pipeline_v2.handlers.indexer import CHUNKS_TABLE, V2IndexPaths, make_index_writer

CORPUS = "test-corpus"


def _embeddings_artifact(store, *, doc_id, fp, corpus=CORPUS, n=2, dims=4):
    body = {
        "doc_id": doc_id,
        "corpus": corpus,
        "content_sha256": "sha",
        "stage_fingerprint": fp,
        "chunks": [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "title": "T",
                "text": f"chunk {i}",
                "page": 1,
                "page_end": 1,
                "section_path": "",
                "source_url": "",
                "vector": [0.1 * (i + 1)] * dims,
            }
            for i in range(n)
        ],
    }
    return store.put(
        json.dumps(body).encode(), key_parts=["embed", corpus, doc_id, fp], suffix=".json"
    )


def _index_doc(tmp_path, *, doc_id, fp, corpus=CORPUS):
    """Write a real Lance/tantivy index entry for doc_id via the production write path."""
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    indexes_root = tmp_path / "indexes"
    writer = make_index_writer(store=store, indexes_root=indexes_root)
    artifact_uri = _embeddings_artifact(store, doc_id=doc_id, fp=fp, corpus=corpus)
    message = StageMessage(
        doc_id=doc_id,
        corpus=corpus,
        content_sha256="sha",
        stage_fingerprint=fp,
        payload={"embeddings_artifact_uri": artifact_uri},
    )
    result = writer(message)
    assert result.state == "ok", result.reason
    return V2IndexPaths.for_corpus(corpus, root=indexes_root)


def _seed_work_state(repo, *, doc_id, fp, state, corpus=CORPUS, stage="indexer"):
    repo.upsert_outcome(
        doc_id=doc_id,
        corpus=corpus,
        stage=stage,
        content_sha256="sha",
        stage_fingerprint=fp,
        state=state,
    )


# --------------------------------------------------------------------------
# The must-fail self-test: proves the oracle can catch the 2,266-doc bug class.
# --------------------------------------------------------------------------


def test_degraded_row_indexed_is_caught(tmp_path, repo):
    """A work_state row marked 'degraded' but present in the Lance index -> FAIL.

    This is exactly the v1 bug: a non-'ok' outcome recorded in one place,
    read back as "done" (i.e. present in the index) somewhere else.
    """
    doc_id, fp = "doc-bug", "fp-bug"
    paths = _index_doc(tmp_path, doc_id=doc_id, fp=fp)
    _seed_work_state(repo, doc_id=doc_id, fp=fp, state="degraded")

    outcome = verify.check_no_dead_letters_indexed(repo, CORPUS, paths.lance_dir)

    assert outcome.status == "FAIL", outcome.detail
    assert doc_id in outcome.detail

    # And the aggregate data-check run must also fail the whole layer.
    outcomes = verify.run_data_checks(
        repo=repo,
        corpus=CORPUS,
        lance_dir=paths.lance_dir,
        bm25_dir=paths.bm25_dir,
        check_dlq=False,
    )
    report = verify.VerifyReport(outcomes=outcomes)
    assert not report.ok
    assert any(
        o.check == "no-dead-letters-in-index" and o.status == "FAIL" for o in report.outcomes
    )


def test_failed_row_indexed_is_caught(tmp_path, repo):
    """Same bug class, state='failed' instead of 'degraded'."""
    doc_id, fp = "doc-bug-2", "fp-bug-2"
    paths = _index_doc(tmp_path, doc_id=doc_id, fp=fp)
    _seed_work_state(repo, doc_id=doc_id, fp=fp, state="failed")

    outcome = verify.check_no_dead_letters_indexed(repo, CORPUS, paths.lance_dir)

    assert outcome.status == "FAIL", outcome.detail
    assert doc_id in outcome.detail


# --------------------------------------------------------------------------
# Happy path: consistent state -> PASS.
# --------------------------------------------------------------------------


def test_happy_path_consistent_state_passes(tmp_path, repo):
    doc_id, fp = "doc-ok", "fp-ok"
    paths = _index_doc(tmp_path, doc_id=doc_id, fp=fp)
    _seed_work_state(repo, doc_id=doc_id, fp=fp, state="ok")

    outcomes = verify.run_data_checks(
        repo=repo,
        corpus=CORPUS,
        lance_dir=paths.lance_dir,
        bm25_dir=paths.bm25_dir,
        check_dlq=False,
    )
    report = verify.VerifyReport(outcomes=outcomes)

    assert report.ok, report.render()
    assert all(o.status in ("PASS", "SKIP") for o in report.outcomes)
    assert any(
        o.check == "ok-rows-present-in-lance" and o.status == "PASS" for o in report.outcomes
    )
    assert any(
        o.check == "no-dead-letters-in-index" and o.status == "PASS" for o in report.outcomes
    )
    assert any(o.check == "no-duplicate-index-rows" and o.status == "PASS" for o in report.outcomes)


def test_happy_path_empty_corpus_passes(tmp_path, repo):
    """No docs at all for this corpus yet -- must not spuriously fail."""
    indexes_root = tmp_path / "indexes"
    paths = V2IndexPaths.for_corpus(CORPUS, root=indexes_root)

    outcomes = verify.run_data_checks(
        repo=repo,
        corpus=CORPUS,
        lance_dir=paths.lance_dir,
        bm25_dir=paths.bm25_dir,
        check_dlq=False,
    )
    report = verify.VerifyReport(outcomes=outcomes)

    assert report.ok, report.render()


# --------------------------------------------------------------------------
# Duplicate-row violation.
# --------------------------------------------------------------------------


def test_duplicate_index_rows_is_caught(tmp_path, repo):
    doc_id, fp = "doc-dup", "fp-dup"
    paths = _index_doc(tmp_path, doc_id=doc_id, fp=fp)
    _seed_work_state(repo, doc_id=doc_id, fp=fp, state="ok")

    # Bypass the indexer's own idempotent delete-then-insert to simulate a
    # duplicate-write bug directly against the Lance table (e.g. a would-be
    # concurrency or corpus-key bug elsewhere in the write path).
    db = lancedb.connect(str(paths.lance_dir))
    table = db.open_table(CHUNKS_TABLE)
    rows = table.to_arrow().to_pylist()
    table.add(rows)  # re-add the exact same rows without deleting first

    outcome = verify.check_no_duplicate_index_rows(paths.lance_dir, CORPUS)

    assert outcome.status == "FAIL", outcome.detail

    outcomes = verify.run_data_checks(
        repo=repo,
        corpus=CORPUS,
        lance_dir=paths.lance_dir,
        bm25_dir=paths.bm25_dir,
        check_dlq=False,
    )
    report = verify.VerifyReport(outcomes=outcomes)
    assert not report.ok


# --------------------------------------------------------------------------
# ok-row-missing-from-index violation (the inverse of the bug class: work_state
# says 'ok' but the index doesn't have it).
# --------------------------------------------------------------------------


def test_ok_row_missing_from_index_is_caught(repo):
    doc_id, fp = "doc-ghost", "fp-ghost"
    _seed_work_state(repo, doc_id=doc_id, fp=fp, state="ok")

    from pathlib import Path

    outcome = verify.check_indexable_present_in_index(
        repo, CORPUS, Path("/nonexistent/lance"), Path("/nonexistent/bm25")
    )[0]

    assert outcome.status == "FAIL", outcome.detail
    assert doc_id in outcome.detail


# --------------------------------------------------------------------------
# V1-integrity layer (no cluster/postgres needed -- pure filesystem).
# --------------------------------------------------------------------------


def test_v1_integrity_no_baseline_is_skip(tmp_path):
    outcome = verify.check_v1_integrity(tmp_path / "does-not-exist.txt")
    assert outcome.status == "SKIP"


def test_v1_integrity_matching_snapshot_passes(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in verify._V1_INGEST_STATE_RELPATHS:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"v": 1}', encoding="utf-8")
    for rel in verify._V1_INDEX_DIR_RELPATHS:
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    snapshot_path = tmp_path / "snapshot.txt"
    verify.write_v1_snapshot(snapshot_path, repo_root=repo_root)

    outcome = verify.check_v1_integrity(snapshot_path, repo_root=repo_root)
    assert outcome.status == "PASS", outcome.detail


def test_v1_integrity_drift_is_caught(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in verify._V1_INGEST_STATE_RELPATHS:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"v": 1}', encoding="utf-8")
    for rel in verify._V1_INDEX_DIR_RELPATHS:
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    snapshot_path = tmp_path / "snapshot.txt"
    verify.write_v1_snapshot(snapshot_path, repo_root=repo_root)

    # Mutate v1 data after the baseline was taken -- the campaign ABORT case.
    (repo_root / verify._V1_INGEST_STATE_RELPATHS[0]).write_text('{"v": 2}', encoding="utf-8")

    outcome = verify.check_v1_integrity(snapshot_path, repo_root=repo_root)
    assert outcome.status == "FAIL"
    assert "ABORT" in outcome.detail


# --------------------------------------------------------------------------
# Report rendering.
# --------------------------------------------------------------------------


def test_report_render_and_ok():
    report = verify.VerifyReport()
    report.add(verify.CheckOutcome("data", "check-a", "PASS", "fine"))
    assert report.ok

    report.add(verify.CheckOutcome("data", "check-b", "FAIL", "bad"))
    assert not report.ok
    rendered = report.render()
    assert "VERDICT: FAIL" in rendered
    assert "check-b" in rendered


# --------------------------------------------------------------------------
# DLQ reconciliation layer, kafka-independent smoke test.
# --------------------------------------------------------------------------


def test_dlq_check_skips_when_kafka_unreachable(repo):
    outcomes = verify.check_dlq_reconciliation(repo, CORPUS, "127.0.0.1:1", timeout=1.0)
    assert len(outcomes) == 1
    assert outcomes[0].status == "SKIP"


@pytest.mark.integration
def test_dlq_check_against_real_kafka_no_failures(repo, kafka_bootstrap):
    """No failed rows for this corpus -> PASS, even with kafka reachable."""
    outcomes = verify.check_dlq_reconciliation(repo, CORPUS, kafka_bootstrap, timeout=10.0)
    assert all(o.status in ("PASS", "SKIP") for o in outcomes), outcomes
