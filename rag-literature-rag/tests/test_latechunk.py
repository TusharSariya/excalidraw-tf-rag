"""Tests for late-chunk window assignment + profile gating (no model load)."""
from __future__ import annotations

from rag_literature_rag.ingest.chunk import TextChunk, assign_late_chunk_windows
from rag_literature_rag.ingest.latechunk import is_latechunk_profile


def _chunk(doc_id: str, idx: int, text: str) -> TextChunk:
    return TextChunk(
        doc_id=doc_id,
        title="t",
        text=text,
        page=1,
        chunk_index=idx,
        source_url="u",
        year=2026,
        tags=[],
        authors=[],
        pipeline_categories=[],
    )


def test_is_latechunk_profile():
    assert is_latechunk_profile("cuda-qwen4b-latechunk-v1")
    assert is_latechunk_profile("foo-LATECHUNK-bar")
    assert not is_latechunk_profile("cuda-qwen4b-1024")
    assert not is_latechunk_profile("cuda-qwen4b-contextual-v1")
    assert not is_latechunk_profile(None)
    assert not is_latechunk_profile("")


def test_window_spans_index_back_to_chunk_text():
    chunks = [_chunk("d1", 0, "alpha alpha"), _chunk("d1", 1, "beta beta beta")]
    assign_late_chunk_windows(chunks, window_chars=10_000)
    # both fit one window
    assert chunks[0].window_id == chunks[1].window_id
    wt = chunks[0].window_text
    for c in chunks:
        assert wt[c.window_char_start:c.window_char_end] == c.text


def test_window_splits_on_size_budget():
    chunks = [_chunk("d1", i, "x" * 100) for i in range(5)]
    assign_late_chunk_windows(chunks, window_chars=250)  # ~2 chunks per window
    wins = {c.window_id for c in chunks}
    assert len(wins) >= 2
    # spans still index correctly within each chunk's own window
    for c in chunks:
        assert c.window_text[c.window_char_start:c.window_char_end] == c.text


def test_windows_never_span_two_documents():
    chunks = [_chunk("d1", 0, "aaa"), _chunk("d2", 0, "bbb")]
    assign_late_chunk_windows(chunks, window_chars=10_000)
    assert chunks[0].window_id != chunks[1].window_id


def test_separator_offset_between_chunks():
    a, b = "alpha", "beta"
    chunks = [_chunk("d1", 0, a), _chunk("d1", 1, b)]
    assign_late_chunk_windows(chunks, window_chars=10_000)
    # second chunk starts after first chunk + "\n\n"
    assert chunks[1].window_char_start == len(a) + 2
    assert chunks[0].window_text == f"{a}\n\n{b}"


def test_index_routes_latechunk_profile_and_bypasses_cache(monkeypatch):
    """A latechunk profile must route through embed_late_chunks and NOT touch
    the per-text embed cache (whose key — chunk text alone — is invalid when the
    vector depends on cross-chunk window context)."""
    from rag_common.config import EmbedConfig
    from rag_literature_rag.ingest import embed_cache, index

    chunks = [_chunk("d1", 0, "alpha alpha"), _chunk("d1", 1, "beta beta")]
    # Native dim for the model is 2560; our fake token vectors are 2-dim, so pin
    # the requested dim to 2 to make pool_span's Matryoshka truncation a no-op.
    cfg = EmbedConfig(
        backend="local",
        model="Qwen/Qwen3-Embedding-4B",
        dimensions=2,
        profile="cuda-qwen4b-latechunk-v1",
    )

    # Stub the heavy token embedder: one 2-dim "token" per char, value = position.
    def fake_tokens(window_texts, *, config, mode="document"):
        out = []
        for wt in window_texts:
            # [i, 1] so the L2-normalized direction varies with token position
            # (a degenerate [i, i] would normalize to the same unit vector).
            vecs = [[float(i), 1.0] for i in range(len(wt))]
            offsets = [(i, i + 1) for i in range(len(wt))]
            out.append((vecs, offsets))
        return out

    monkeypatch.setattr(
        "rag_literature_rag.ingest.latechunk.embed_local_texts_with_tokens",
        fake_tokens,
    )
    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("embed cache must be bypassed for latechunk")

    monkeypatch.setattr(embed_cache, "get", _boom)
    monkeypatch.setattr(embed_cache, "put", _boom)

    vectors = index._embed_index_texts(
        chunks,
        texts=["ignored", "ignored"],
        titles=None,
        cfg=cfg,
        stats=None,
        workers=None,
        phase_stats=None,
    )
    assert len(vectors) == len(chunks)
    assert all(len(v) == 2 for v in vectors)
    # Distinct spans within the window → distinct pooled vectors.
    assert vectors[0] != vectors[1]
