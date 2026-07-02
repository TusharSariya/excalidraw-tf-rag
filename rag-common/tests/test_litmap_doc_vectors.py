from __future__ import annotations

import math

from rag_common.litmap import doc_vectors as dv


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def test_mean_pool_zero_chunks_yields_no_entry():
    out = dv.mean_pool_chunks([])
    assert out == {}


def test_mean_pool_one_chunk_normalizes():
    out = dv.mean_pool_chunks([("d1", [3.0, 4.0])])
    assert set(out) == {"d1"}
    assert math.isclose(_norm(out["d1"]), 1.0, abs_tol=1e-9)
    # direction preserved: 3:4 ratio -> 0.6:0.8
    assert math.isclose(out["d1"][0], 0.6, abs_tol=1e-9)
    assert math.isclose(out["d1"][1], 0.8, abs_tol=1e-9)


def test_mean_pool_many_chunks_averages_then_normalizes():
    chunks = [("d1", [1.0, 0.0]), ("d1", [1.0, 0.0]), ("d1", [0.0, 1.0])]
    out = dv.mean_pool_chunks(chunks)
    # raw mean = (0.667, 0.333) -> normalized
    assert math.isclose(_norm(out["d1"]), 1.0, abs_tol=1e-9)
    assert out["d1"][0] > out["d1"][1]  # two votes for axis 0, one for axis 1


def test_mean_pool_multi_doc_isolated():
    chunks = [("d1", [1.0, 0.0]), ("d2", [0.0, 1.0])]
    out = dv.mean_pool_chunks(chunks)
    assert set(out) == {"d1", "d2"}
    assert math.isclose(out["d1"][0], 1.0, abs_tol=1e-9)
    assert math.isclose(out["d2"][1], 1.0, abs_tol=1e-9)


def test_l2_normalize_zero_vector_is_noop():
    assert dv.l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_save_and_load_doc_vectors_round_trip(tmp_path):
    vectors = {"d1": [0.6, 0.8], "d2": [1.0, 0.0]}
    dv.save_doc_vectors(tmp_path, vectors, embed_profile="p1", index_fingerprint="fp1")
    loaded = dv.load_doc_vectors(tmp_path, embed_profile="p1", index_fingerprint="fp1")
    assert loaded is not None
    assert set(loaded) == {"d1", "d2"}
    for doc_id in vectors:
        assert math.isclose(loaded[doc_id][0], vectors[doc_id][0], abs_tol=1e-5)
        assert math.isclose(loaded[doc_id][1], vectors[doc_id][1], abs_tol=1e-5)


def test_load_doc_vectors_stale_fingerprint_is_a_miss(tmp_path):
    vectors = {"d1": [1.0, 0.0]}
    dv.save_doc_vectors(tmp_path, vectors, embed_profile="p1", index_fingerprint="fp1")
    assert dv.load_doc_vectors(tmp_path, embed_profile="p1", index_fingerprint="fp2") is None
    assert dv.load_doc_vectors(tmp_path, embed_profile="p2", index_fingerprint="fp1") is None


def test_load_doc_vectors_missing_dir_is_a_miss(tmp_path):
    assert (
        dv.load_doc_vectors(tmp_path / "nope", embed_profile="p1", index_fingerprint="fp1") is None
    )


def test_save_doc_vectors_meta_has_schema_version(tmp_path):
    dv.save_doc_vectors(tmp_path, {"d1": [1.0, 0.0]}, embed_profile="p1", index_fingerprint="fp1")
    meta = dv.load_doc_vectors_meta(tmp_path)
    assert meta is not None
    assert meta["schema_version"] == 1
    assert meta["doc_count"] == 1
    assert meta["dim"] == 2
