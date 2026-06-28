"""TLDR-prefix chunk augmentation (T9 A/B arm).

Validates the profile gate and the prepend behavior (per-doc TLDR shared across its
chunks, fail-soft when a doc has no TLDR) without touching a real index or store.
"""
from __future__ import annotations

import graph_layout_rag.ingest.tldr_prefix as tp
from graph_layout_rag.ingest.chunk import TextChunk
from graph_layout_rag.ingest.tldr_prefix import augment_texts_for_tldr, is_tldr_profile


def _chunk(doc_id: str, text: str, idx: int = 0) -> TextChunk:
    return TextChunk(
        doc_id=doc_id,
        title="T",
        text=text,
        page=None,
        chunk_index=idx,
        source_url="",
        year=None,
        tags=[],
        authors=[],
        pipeline_categories=[],
    )


# ----------------------------------------------------------------- profile gate
def test_is_tldr_profile():
    assert is_tldr_profile("cuda-qwen0.6b-tldr-v1")
    assert is_tldr_profile("TLDR-experiment")
    assert not is_tldr_profile("cuda-qwen0.6b-1024")
    assert not is_tldr_profile("cuda-qwen0.6b-section-v1")
    assert not is_tldr_profile(None)


# ----------------------------------------------------------------- augmentation
def test_prepends_tldr_per_doc(monkeypatch):
    # d1 has a TLDR (shared by both its chunks); d2 has none → left unchanged.
    monkeypatch.setattr(tp, "_tldr_by_doc", lambda ids: {"d1": "Layered drawings minimize crossings."})
    chunks = [_chunk("d1", "body-a", 0), _chunk("d1", "body-b", 1), _chunk("d2", "body-c", 0)]
    texts = ["body-a", "body-b", "body-c"]
    out = augment_texts_for_tldr(chunks, texts)
    assert out[0] == "TLDR: Layered drawings minimize crossings.\nbody-a"
    assert out[1] == "TLDR: Layered drawings minimize crossings.\nbody-b"
    assert out[2] == "body-c"  # no TLDR for d2 → unchanged


def test_no_tldrs_returns_texts_unchanged(monkeypatch):
    monkeypatch.setattr(tp, "_tldr_by_doc", lambda ids: {})
    chunks = [_chunk("d1", "x"), _chunk("d2", "y")]
    texts = ["x", "y"]
    out = augment_texts_for_tldr(chunks, texts)
    assert out == texts
    assert out is texts  # exact passthrough, no allocation


def test_missing_store_yields_empty_map(monkeypatch):
    import graph_layout_rag.paths as paths

    class _P:
        @staticmethod
        def exists() -> bool:
            return False

    monkeypatch.setattr(paths, "CITATIONS_DB_PATH", _P)
    assert tp._tldr_by_doc({"d1"}) == {}
