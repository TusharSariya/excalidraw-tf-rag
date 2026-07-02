from __future__ import annotations

from rag_common.litmap import neighborhood as nb
from rag_common.litmap.types import CitationAdjacency


def _adj(out_adj, in_adj, doc_to_oa):
    undirected: dict[str, set[str]] = {}
    for a, outs in out_adj.items():
        undirected.setdefault(a, set()).update(outs)
        for b in outs:
            undirected.setdefault(b, set()).add(a)
    for a, ins in in_adj.items():
        undirected.setdefault(a, set()).update(ins)
        for b in ins:
            undirected.setdefault(b, set()).add(a)
    return CitationAdjacency(
        out_adj=out_adj,
        in_adj=in_adj,
        undirected=undirected,
        doc_to_oa=doc_to_oa,
        oa_to_doc={v: k for k, v in doc_to_oa.items()},
    )


def test_neighborhood_graph_blends_signals_and_respects_top_n():
    doc_to_oa = {"seed": "S", "c1": "C1", "c2": "C2", "c3": "C3"}
    out_adj = {"S": {"R1"}, "C1": {"R1"}, "C2": {"R1"}, "C3": set()}
    in_adj = {"S": {"X1"}, "C1": {"X1"}, "C2": set(), "C3": {"X1"}}
    adj = _adj(out_adj, in_adj, doc_to_oa)
    graph = nb.neighborhood_graph("seed", ["c1", "c2", "c3"], adj, top_n=2)
    assert graph["schema_version"] == 1
    assert graph["seed"] == "seed"
    assert graph["nodes"][0] == "seed"
    assert len(graph["nodes"]) - 1 <= 2
    # c1 shares both coupling and co-citation with seed -> should rank first.
    assert graph["edges"][0]["dst"] == "c1"


def test_neighborhood_graph_empty_when_no_signal():
    doc_to_oa = {"seed": "S", "c1": "C1"}
    adj = _adj({"S": set()}, {"S": set()}, doc_to_oa)
    graph = nb.neighborhood_graph("seed", ["c1"], adj)
    assert graph["nodes"] == ["seed"]
    assert graph["edges"] == []


def test_to_dot_renders_nodes_and_edges():
    graph = {
        "nodes": ["seed", "c1"],
        "edges": [{"src": "seed", "dst": "c1", "weight": 0.5}],
    }
    dot = nb.to_dot(graph)
    assert dot.startswith("graph G {")
    assert '"seed"' in dot
    assert '"seed" -- "c1"' in dot
    assert dot.strip().endswith("}")
