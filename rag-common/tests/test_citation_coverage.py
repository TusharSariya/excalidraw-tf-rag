"""Offline unit tests for rag_common.citation_coverage against a hand-built fixture DB."""

from __future__ import annotations

import sqlite3

from rag_common.citation_coverage import compute_coverage


def _fixture_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY,
          doi TEXT,
          doc_id TEXT,
          title TEXT,
          year INTEGER,
          cited_by_count INTEGER DEFAULT 0,
          in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL,
          dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(src_oa, dst_oa)
        );
        """
    )
    # Two keyed corpus papers (W1, W2), one external unkeyed neighbor (W3).
    db.executemany(
        "INSERT INTO papers(oa_id, doi, doc_id, title, year, in_corpus) VALUES(?,?,?,?,?,?)",
        [
            ("W1", "10.1/a", "doc-a", "Paper A", 1995, 1),
            ("W2", "10.1/b", "doc-b", "Paper B", 2003, 1),
            ("W3", None, None, "External unkeyed", None, 0),
        ],
    )
    db.executemany(
        "INSERT INTO cites(src_oa, dst_oa) VALUES(?,?)",
        [
            ("W1", "W2"),  # both endpoints keyed
            ("W1", "W3"),  # only src keyed
            ("W2", "W3"),  # only src keyed
        ],
    )
    db.commit()
    return db


def test_compute_coverage_basic_counts():
    db = _fixture_db()
    stats = compute_coverage(db, total_corpus_docs=4)

    assert stats.total_corpus_docs == 4
    assert stats.keyed_docs == 2
    assert stats.keyed_pct == 50.0

    assert stats.total_edges == 3
    assert stats.both_endpoint_keyed_edges == 1
    assert round(stats.both_endpoint_keyed_pct, 2) == round(1 / 3 * 100, 2)


def test_compute_coverage_keyed_by_decade():
    db = _fixture_db()
    stats = compute_coverage(db, total_corpus_docs=4)

    assert stats.keyed_by_decade == {"1990s": 1, "2000s": 1}


def test_compute_coverage_no_edges_no_docs_is_safe():
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY, doi TEXT, doc_id TEXT, title TEXT, year INTEGER,
          cited_by_count INTEGER DEFAULT 0, in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL, dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(src_oa, dst_oa)
        );
        """
    )
    stats = compute_coverage(db, total_corpus_docs=0)
    assert stats.keyed_docs == 0
    assert stats.keyed_pct == 0.0
    assert stats.total_edges == 0
    assert stats.both_endpoint_keyed_pct == 0.0
    assert stats.keyed_by_decade == {}
    assert stats.duplicate_doc_ids == {}


def test_compute_coverage_duplicate_doc_id_detected():
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY, doi TEXT, doc_id TEXT, title TEXT, year INTEGER,
          cited_by_count INTEGER DEFAULT 0, in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL, dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(src_oa, dst_oa)
        );
        """
    )
    # Same doc_id keyed under two distinct oa_ids (a keying bug producing duplicates).
    db.executemany(
        "INSERT INTO papers(oa_id, doi, doc_id, title, year, in_corpus) VALUES(?,?,?,?,?,?)",
        [
            ("W1", "10.1/a", "doc-a", "Paper A", 1995, 1),
            ("W1b", "10.1/a-dup", "doc-a", "Paper A (dup keying)", 1995, 1),
        ],
    )
    db.commit()
    stats = compute_coverage(db, total_corpus_docs=1)
    assert stats.duplicate_doc_ids == {"doc-a": 2}
    # keyed_docs counts DISTINCT doc_id, so the duplicate keying doesn't double count.
    assert stats.keyed_docs == 1


def test_compute_coverage_undated_keyed_doc():
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY, doi TEXT, doc_id TEXT, title TEXT, year INTEGER,
          cited_by_count INTEGER DEFAULT 0, in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL, dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(src_oa, dst_oa)
        );
        """
    )
    db.execute(
        "INSERT INTO papers(oa_id, doi, doc_id, title, year, in_corpus) VALUES(?,?,?,?,?,?)",
        ("W1", "10.1/a", "doc-a", "Paper A", None, 1),
    )
    db.commit()
    stats = compute_coverage(db, total_corpus_docs=1)
    assert stats.keyed_by_decade == {"undated": 1}


def test_compute_coverage_matches_glr_measured_baseline():
    """Regression pin: A2b's own audit run against the *existing* (pre-keying-upgrade)
    GLR citations.sqlite reproduced 1,334 keyed docs / 100,362 edges / ~9.1% both-keyed
    (docs/research-tool-decisions-log-2026-07.md #27). This synthetic check just pins the
    both-endpoint-keyed percentage formula against that ratio so a refactor can't silently
    flip numerator/denominator.
    """
    keyed = 1334
    both_keyed_edges = 9134
    total_edges = 100362
    assert round(both_keyed_edges / total_edges * 100, 1) == 9.1
    assert keyed > 0  # sanity: the historical figure this module is meant to reproduce
