from __future__ import annotations

from rag_common.litmap import similarity as sim
from rag_common.litmap.types import CitationAdjacency, Neighbor


def _adj(out_adj, in_adj, doc_to_oa):
    undirected: dict[str, set[str]] = {}
    for a, outs in out_adj.items():
        undirected.setdefault(a, set()).update(outs)
        for b in outs:
            undirected.setdefault(b, set()).add(a)
    return CitationAdjacency(
        out_adj=out_adj,
        in_adj=in_adj,
        undirected=undirected,
        doc_to_oa=doc_to_oa,
        oa_to_doc={v: k for k, v in doc_to_oa.items()},
    )


def test_bib_coupling_no_refs_is_empty():
    # seed has no outgoing references at all.
    adj = _adj(out_adj={"S": set()}, in_adj={}, doc_to_oa={"seed": "S", "c1": "C1"})
    assert sim.bib_coupling_neighbors("seed", ["c1"], adj) == []


def test_bib_coupling_isolated_candidate_is_excluded():
    adj = _adj(
        out_adj={"S": {"R1"}, "C1": set()},
        in_adj={},
        doc_to_oa={"seed": "S", "c1": "C1"},
    )
    assert sim.bib_coupling_neighbors("seed", ["c1"], adj) == []


def test_bib_coupling_shared_refs_ranked():
    out_adj = {
        "S": {"R1", "R2"},
        "C1": {"R1"},  # shares 1 ref
        "C2": {"R1", "R2"},  # shares 2 refs
    }
    doc_to_oa = {"seed": "S", "c1": "C1", "c2": "C2"}
    adj = _adj(out_adj=out_adj, in_adj={}, doc_to_oa=doc_to_oa)
    result = sim.bib_coupling_neighbors("seed", ["c1", "c2"], adj)
    assert [n.doc_id for n in result] == ["c2", "c1"]
    assert result[0].rank == 1 and result[1].rank == 2


def test_co_citation_no_citers_is_empty():
    adj = _adj(out_adj={}, in_adj={"S": set()}, doc_to_oa={"seed": "S", "c1": "C1"})
    assert sim.co_citation_neighbors("seed", ["c1"], adj) == []


def test_cosine_knn_ranks_by_similarity():
    vectors = {
        "seed": [1.0, 0.0],
        "close": [0.9, 0.1],
        "far": [0.0, 1.0],
    }
    result = sim.cosine_knn("seed", vectors, top_k=5)
    assert [n.doc_id for n in result] == ["close", "far"]


def test_cosine_knn_missing_seed_is_empty():
    assert sim.cosine_knn("missing", {"a": [1.0, 0.0]}) == []


def test_rrf_fuse_alpha_zero_reduces_to_text_only():
    text = [Neighbor("a", 1, 0.9), Neighbor("b", 2, 0.5), Neighbor("c", 3, 0.1)]
    citation = [Neighbor("z", 1, 5.0), Neighbor("y", 2, 3.0)]
    fused = sim.fuse_neighbors(text, citation, alpha=0.0)
    assert [n.doc_id for n in fused] == ["a", "b", "c"]


def test_rrf_fuse_alpha_one_reduces_to_citation_only():
    text = [Neighbor("a", 1, 0.9), Neighbor("b", 2, 0.5)]
    citation = [Neighbor("z", 1, 5.0), Neighbor("y", 2, 3.0), Neighbor("x", 3, 1.0)]
    fused = sim.fuse_neighbors(text, citation, alpha=1.0)
    assert [n.doc_id for n in fused] == ["z", "y", "x"]


def test_rrf_fuse_blends_both_signals():
    # doc "shared" appears in both signals near the top -> should outrank a doc that
    # only appears in one signal, even at alpha=0.5.
    text = [Neighbor("shared", 1, 0.9), Neighbor("text_only", 2, 0.5)]
    citation = [Neighbor("shared", 1, 5.0), Neighbor("cite_only", 2, 3.0)]
    fused = sim.fuse_neighbors(text, citation, alpha=0.5)
    assert fused[0].doc_id == "shared"


def test_fuse_neighbors_rejects_out_of_range_alpha():
    import pytest

    with pytest.raises(ValueError):
        sim.fuse_neighbors([], [], alpha=1.5)
