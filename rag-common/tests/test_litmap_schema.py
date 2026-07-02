"""JSON schema snapshot: every litmap output dict carries schema_version=1 plus its
documented required keys. Guards against an accidental key rename/removal breaking the
CLI's `--json` contract silently.
"""

from __future__ import annotations

import pytest
from rag_common.litmap import mainpath as mp
from rag_common.litmap import neighborhood as nb
from rag_common.litmap.types import CitationAdjacency

igraph = pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from rag_common.litmap import communities as com  # noqa: E402


def test_communities_schema():
    result = com.build_clusters(
        [("a", "b", 1.0)],
        {"a": {"title": "A", "tags": []}, "b": {"title": "B", "tags": []}},
    )
    assert result["schema_version"] == 1
    for key in ("cluster_count", "doc_count", "clusters"):
        assert key in result
    for cluster in result["clusters"]:
        for key in ("id", "size", "label", "label_method", "doc_ids"):
            assert key in cluster


def test_mainpath_schema():
    result = mp.build_main_path([("a", "b"), ("b", "c")])
    assert result["schema_version"] == 1
    for key in ("main_path", "edges", "node_count", "edge_count", "warnings"):
        assert key in result


def test_neighborhood_schema():
    adj = CitationAdjacency(
        out_adj={"S": {"R1"}, "C1": {"R1"}},
        in_adj={},
        undirected={"S": {"R1"}, "R1": {"S", "C1"}, "C1": {"R1"}},
        doc_to_oa={"seed": "S", "c1": "C1"},
        oa_to_doc={"S": "seed", "C1": "c1"},
    )
    graph = nb.neighborhood_graph("seed", ["c1"], adj)
    assert graph["schema_version"] == 1
    for key in ("seed", "nodes", "edges"):
        assert key in graph
