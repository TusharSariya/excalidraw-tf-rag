"""Extract stage reuses v1's cached docling chunks so v2 chunk_index /
canonical_sha256 match v1 (the precondition for the contextual + extract caches
to hit). Verified against a stubbed corpus extract_cache module."""

from __future__ import annotations

import sys
import types

from pipeline_v2.handlers import extract as ex


def test_load_cached_docling_chunks_maps_v1_textchunks(monkeypatch):
    # Stub a corpus package's ingest.extract_cache + ingest.chunk so the reuse
    # path resolves without the real v1 tool installed.
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

    ec = types.ModuleType("stubpkg.ingest.extract_cache")
    ec.get = lambda sha, backend, profile: (
        [_TC(0), _TC(1)] if backend == "docling" and sha == "abc" else None
    )
    ck = types.ModuleType("stubpkg.ingest.chunk")
    ck.embed_input_text = lambda c: f"EMBED::{c.text}"
    for name, mod in {
        "stubpkg": types.ModuleType("stubpkg"),
        "stubpkg.ingest": types.ModuleType("stubpkg.ingest"),
        "stubpkg.ingest.extract_cache": ec,
        "stubpkg.ingest.chunk": ck,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(ex._CORPUS_PKG, "glr", "stubpkg")

    recs = ex._load_cached_docling_chunks("glr", "abc")
    assert recs is not None and len(recs) == 2
    assert [r["chunk_index"] for r in recs] == [0, 1]
    assert recs[0]["embed_text"] == "EMBED::body 0"  # embed_input_text applied
    assert recs[1]["section_path"] == "Intro"
    # miss → None (caller falls back to PyMuPDF)
    assert ex._load_cached_docling_chunks("glr", "nope") is None
    assert ex._load_cached_docling_chunks("unknown-corpus", "abc") is None
