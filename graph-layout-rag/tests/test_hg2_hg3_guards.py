"""HG2 (chunk-count parity) + HG3 (synced-BM25 integrity) guards.

A half-failed or resumed reembed can leave a rung short while `reembed_completed_at` is
still written; a truncated rsync can leave the desktop BM25 index divergent from the
Mac's. Both are silent-and-unguarded without these checks.
"""

import hashlib

import pytest

from graph_layout_rag.ingest import guards


def test_assert_chunk_parity_matches(monkeypatch):
    monkeypatch.setattr(guards, "chunk_count", lambda p: 41083)
    assert guards.assert_chunk_parity("src", "tgt") == 41083


def test_assert_chunk_parity_short_target_raises(monkeypatch):
    counts = {"src": 41083, "tgt": 40000}
    monkeypatch.setattr(guards, "chunk_count", lambda p: counts[p])
    with pytest.raises(ValueError, match="HG2 parity FAIL"):
        guards.assert_chunk_parity("src", "tgt")


def test_assert_chunk_parity_empty_source_raises(monkeypatch):
    monkeypatch.setattr(guards, "chunk_count", lambda p: 0)
    with pytest.raises(ValueError, match="0 chunks"):
        guards.assert_chunk_parity("src", "tgt")


def test_compare_fingerprints_identical():
    fp = {
        "chunk_count": 41083,
        "bm25_doc_count": 41083,
        "bm25_tree_hash": "abc",
        "bm25_n_files": 19890,
    }
    assert guards.compare_fingerprints(fp, dict(fp)) == []


def test_compare_fingerprints_detects_truncated_sync():
    local = {"chunk_count": 41083, "bm25_doc_count": 41083, "bm25_tree_hash": "abc", "bm25_n_files": 19890}
    remote = {"chunk_count": 41083, "bm25_doc_count": 40000, "bm25_tree_hash": "xyz", "bm25_n_files": 19000}
    problems = guards.compare_fingerprints(local, remote)
    assert any("bm25_doc_count" in p for p in problems)
    assert any("bm25_tree_hash" in p for p in problems)


def test_bm25_tree_hash_excludes_host_specific_and_locks(tmp_path):
    d = tmp_path / "bm25"
    d.mkdir()
    (d / "seg.idx").write_bytes(b"segment-data")
    (d / "meta.json").write_text("host-specific")
    (d / ".managed.json").write_text("host-specific")
    (d / ".tantivy.lock").write_bytes(b"")
    h1, n1 = guards._bm25_tree_hash(d)
    # Mutating excluded files must NOT change the hash (they are not byte-stable on copy).
    (d / "meta.json").write_text("different-host")
    (d / ".managed.json").write_text("different")
    h2, n2 = guards._bm25_tree_hash(d)
    assert h1 == h2 and n1 == n2 == 1
    # Mutating a real segment file MUST change the hash.
    (d / "seg.idx").write_bytes(b"corrupted")
    h3, _ = guards._bm25_tree_hash(d)
    assert h3 != h1
