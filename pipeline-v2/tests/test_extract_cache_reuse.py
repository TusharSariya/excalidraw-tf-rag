"""Extract stage: PDF backend is a swappable flag (docling/pymupdf), and the
chosen backend selects the extract_cache namespace so v2 reuses v1's chunks
(matching chunk_index / canonical_sha256). Covers the pure helpers."""

from __future__ import annotations

from pipeline_v2.handlers import extract as ex


def test_pdf_backend_flag(monkeypatch):
    monkeypatch.delenv("PIPELINE_V2_PDF_BACKEND", raising=False)
    assert ex._pdf_backend() == "pymupdf"  # default
    monkeypatch.setenv("PIPELINE_V2_PDF_BACKEND", "docling")
    assert ex._pdf_backend() == "docling"
    monkeypatch.setenv("PIPELINE_V2_PDF_BACKEND", "DOCLING")
    assert ex._pdf_backend() == "docling"  # case-insensitive
    monkeypatch.setenv("PIPELINE_V2_PDF_BACKEND", "bogus")
    assert ex._pdf_backend() == "pymupdf"  # invalid → default


def test_records_from_chunks_maps_fields():
    class _TC:
        def __init__(self, i):
            self.doc_id = "d1"
            self.chunk_index = i
            self.title = "T"
            self.text = f"body {i}"
            self.page = i
            self.page_end = i
            self.section_path = "Intro"
            self.source_url = "http://x"

    recs = ex._records_from_chunks([_TC(0), _TC(1)], lambda c: f"EMBED::{c.text}")
    assert [r["chunk_index"] for r in recs] == [0, 1]
    assert recs[0]["embed_text"] == "EMBED::body 0"  # embed_input_text applied
    assert recs[1]["section_path"] == "Intro"
    assert recs[0]["doc_id"] == "d1"
