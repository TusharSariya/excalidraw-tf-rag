"""Offline tests for the learned-sparse index + cache + retrieval fusion.

No models, no network: a deterministic fake encoder produces the ``{token_id:
weight}`` bags, so the whole learned-sparse path (build -> load -> score ->
fuse) is exercised on a bare runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rag_literature_rag.ingest import sparse_cache, sparse_index
from rag_literature_rag.query.hybrid import reciprocal_rank_fusion

MODEL_ID = "fake/sparse"
VERSION = "fake-1"


def mk_chunk(doc_id: str, chunk_index: int, *, year: int | None = 2020) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title=f"Title {doc_id}",
        tags=["t1", "t2"],
        pipeline_categories=["layering"],
        section_path="Intro",
        alias_doc_ids=[],
        alias_source_urls=[],
        alias_dois=[],
        canonical_sha256="sha",
        source_url="http://example/" + doc_id,
        year=year,
        page=1,
        page_end=2,
    )


class CountingFakeEncoder:
    model_id = MODEL_ID
    version = VERSION

    def __init__(self) -> None:
        self.doc_calls = 0

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        self.doc_calls += len(texts)
        return [self.encode_query(t) for t in texts]

    def encode_query(self, text: str) -> dict[int, float]:
        bag: dict[int, float] = {}
        for i, token in enumerate(text.lower().split()):
            tid = (abs(hash(token)) % 5000) + 1
            bag[tid] = bag.get(tid, 0.0) + 1.0 + i * 0.1
        return bag


def _build(index_dir, texts, *, encoder=None, rebuild=False):
    encoder = encoder or CountingFakeEncoder()
    chunks = [mk_chunk(f"d{i}", 0) for i in range(len(texts))]
    tokens = encoder.encode_documents(texts)
    n = sparse_index.upsert_chunks(
        chunks,
        texts,
        tokens,
        index_dir=index_dir,
        model_id=encoder.model_id,
        version=encoder.version,
        rebuild=rebuild,
    )
    return n, chunks, tokens, encoder


def test_build_save_load_roundtrip(tmp_path):
    idx = tmp_path / "sparse"
    texts = ["stress majorization layout", "sugiyama layering barycenter", "graphviz dot ranks"]
    n, chunks, _, _ = _build(idx, texts)
    assert n == 3
    meta = sparse_index.load_meta(idx)
    assert meta is not None
    assert meta.model_id == MODEL_ID and meta.version == VERSION
    assert meta.chunk_count == 3
    assert len(meta.checksum) == 16  # 64-bit hex
    assert sparse_index.chunk_count(idx) == 3


def test_scoring_matches_bruteforce(tmp_path):
    idx = tmp_path / "sparse"
    texts = [
        "stress majorization layout",
        "sugiyama layering barycenter crossing",
        "graphviz dot ranks layout",
        "force directed layout stress",
    ]
    _, chunks, tokens, enc = _build(idx, texts)
    query = "stress layout"
    q = enc.encode_query(query)

    brute: dict[str, float] = {}
    for chunk, bag in zip(chunks, tokens):
        s = sum(q[t] * w for t, w in bag.items() if t in q)
        if s > 0:
            brute[f"{chunk.doc_id}:{chunk.chunk_index}"] = round(s, 6)

    got = sparse_index.score(q, index_dir=idx, limit=10)
    got_scores = {r["id"]: r["score"] for r in got}
    assert got_scores == brute
    # Descending order.
    scores = [r["score"] for r in got]
    assert scores == sorted(scores, reverse=True)
    # Candidate rows carry the BM25-shaped fields used downstream.
    for r in got:
        assert set(r) >= {"id", "doc_id", "title", "text", "tags", "score", "section_path"}


def test_learned_leg_fuses_like_tantivy(tmp_path):
    idx = tmp_path / "sparse"
    texts = ["alpha beta", "beta gamma", "gamma delta"]
    _, _, _, enc = _build(idx, texts)
    sparse_rows = sparse_index.score(enc.encode_query("beta gamma"), index_dir=idx, limit=10)
    assert sparse_rows  # non-empty
    dense_rows = [{"id": "d0:0", "score": 0.9}, {"id": "d2:0", "score": 0.8}]
    fused = reciprocal_rank_fusion(dense_rows, sparse_rows, top=10)
    ids = {r["id"] for r in fused}
    assert {"d0:0", "d1:0", "d2:0"} <= ids
    assert all("fusion_score" in r for r in fused)
    # A learned-only hit (d1:0, absent from dense) still surfaces via the sparse leg.
    assert any(r["id"] == "d1:0" for r in fused)


def test_cache_incrementality(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_cache, "_CACHE_DIR", tmp_path / "sparse_cache")
    enc = CountingFakeEncoder()
    texts = ["one two three", "four five six"]
    first = sparse_cache.encode_documents_cached(texts, encoder=enc)
    assert enc.doc_calls == 2
    # Second pass over the same texts: all cache hits, no re-encode.
    second = sparse_cache.encode_documents_cached(texts, encoder=enc)
    assert enc.doc_calls == 2
    assert first == second
    # A new text encodes only the miss.
    sparse_cache.encode_documents_cached([*texts, "seven eight"], encoder=enc)
    assert enc.doc_calls == 3


def test_metadata_guard_refuses_mismatch(tmp_path):
    idx = tmp_path / "sparse"
    _build(idx, ["alpha beta", "gamma"])
    good = SimpleNamespace(model_id=MODEL_ID, version=VERSION)
    sparse_index.assert_encoder_matches(idx, good)  # no raise
    bad_model = SimpleNamespace(model_id="other/model", version=VERSION)
    with pytest.raises(RuntimeError, match="mismatched sparse model"):
        sparse_index.assert_encoder_matches(idx, bad_model)
    bad_version = SimpleNamespace(model_id=MODEL_ID, version="fake-2")
    with pytest.raises(RuntimeError, match="mismatched sparse model"):
        sparse_index.assert_encoder_matches(idx, bad_version)
    # No index at all -> refuse rather than silently return nothing.
    with pytest.raises(RuntimeError, match="No learned-sparse index"):
        sparse_index.assert_encoder_matches(tmp_path / "absent", good)


def test_upsert_replaces_and_rebuild(tmp_path):
    idx = tmp_path / "sparse"
    enc = CountingFakeEncoder()
    chunks = [mk_chunk("d0", 0)]
    sparse_index.upsert_chunks(
        chunks,
        ["alpha beta"],
        enc.encode_documents(["alpha beta"]),
        index_dir=idx,
        model_id=MODEL_ID,
        version=VERSION,
    )
    # Re-upsert same chunk id with new content -> still one chunk, new postings.
    sparse_index.upsert_chunks(
        chunks,
        ["gamma delta"],
        enc.encode_documents(["gamma delta"]),
        index_dir=idx,
        model_id=MODEL_ID,
        version=VERSION,
    )
    assert sparse_index.chunk_count(idx) == 1
    assert sparse_index.score(enc.encode_query("alpha"), index_dir=idx, limit=5) == []
    assert sparse_index.score(enc.encode_query("gamma"), index_dir=idx, limit=5)
    # A different encoder identity is refused without rebuild.
    with pytest.raises(RuntimeError, match="re-run with --rebuild"):
        sparse_index.upsert_chunks(
            chunks,
            ["x"],
            [{1: 1.0}],
            index_dir=idx,
            model_id="other",
            version="v9",
        )
    # rebuild wipes and accepts the new identity.
    sparse_index.upsert_chunks(
        chunks,
        ["x y"],
        [{1: 1.0, 2: 2.0}],
        index_dir=idx,
        model_id="other",
        version="v9",
        rebuild=True,
    )
    meta = sparse_index.load_meta(idx)
    assert meta.model_id == "other" and meta.chunk_count == 1
