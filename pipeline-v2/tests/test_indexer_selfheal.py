"""Indexer BM25 crash-recovery: a tantivy pod-kill mid-merge leaves a missing
segment; the indexer must self-heal by rebuilding from LanceDB, not dead-letter
every subsequent doc forever (found during the E15 full-ingest chaos run)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline_v2.handlers import indexer as ix


def _row(i: int, doc_id: str = "d1") -> dict:
    return {
        "id": f"{doc_id}-{i}", "doc_id": doc_id, "corpus": "glr", "title": "t",
        "text": f"hello world chunk {i}", "section_path": "", "source_url": "",
        "chunk_index": i, "page": 0, "vector": [0.1, 0.2],
    }


def test_is_bm25_corruption_detects_filedoesnotexist():
    assert ix._is_bm25_corruption(Exception("Failed to open file for read: 'FileDoesNotExist(\"/x/bm25/ab.term\")'"))
    assert not ix._is_bm25_corruption(Exception("some unrelated error"))


def test_write_bm25_self_heals_from_corrupted_index(tmp_path: Path):
    bm25 = tmp_path / "bm25"
    lance = tmp_path / "lancedb"
    # Seed lance + a healthy bm25 with two docs.
    rows_d1 = [_row(0, "d1"), _row(1, "d1")]
    ix._write_lance(lance, "d1", "glr", rows_d1)
    ix._write_bm25(bm25, "d1", "glr", rows_d1, lance_dir=lance)
    assert ix.bm25_doc_ids(bm25) == {"d1"}
    # Corrupt the tantivy index: delete a segment file so the next open/commit
    # raises FileDoesNotExist (simulates a kill mid-merge).
    seg = next((p for p in bm25.iterdir() if p.suffix in (".term", ".idx", ".store", ".pos", ".fast", ".fieldnorm")), None)
    assert seg is not None, "expected a tantivy segment file to corrupt"
    seg.unlink()
    # Add a new doc to lance, then write its bm25 — must self-heal (rebuild from
    # lance, which now has d1 + d2) instead of raising.
    rows_d2 = [_row(0, "d2")]
    ix._write_lance(lance, "d2", "glr", rows_d2)
    ix._write_bm25(bm25, "d2", "glr", rows_d2, lance_dir=lance)
    # After self-heal the index is consistent and holds BOTH docs (rebuilt from lance).
    assert ix.bm25_doc_ids(bm25) == {"d1", "d2"}


def test_write_bm25_reraises_non_corruption(tmp_path: Path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("totally unrelated")
    monkeypatch.setattr(ix, "_open_or_create_bm25", boom)
    with pytest.raises(ValueError, match="unrelated"):
        ix._write_bm25(tmp_path / "b", "d", "glr", [_row(0)], lance_dir=tmp_path / "l")
