from graph_layout_rag.catalog.classify import classify_item
from graph_layout_rag.catalog.taxonomy import categories_from_tags
from graph_layout_rag.manifest import ManifestItem


def _item(**kwargs) -> ManifestItem:
    defaults = {
        "id": "test-doc",
        "title": "Untitled",
        "source": "test",
        "url": "https://example.com",
        "status": "ok",
    }
    defaults.update(kwargs)
    return ManifestItem(**defaults)


def test_sander_compound_by_tag():
    item = _item(
        id="sander-compound-directed-graphs",
        title="Layout of Compound Directed Graphs",
        tags=["compound", "layered", "dagre", "clustering"],
    )
    categories, methods = classify_item(item)
    assert "compound" in categories
    assert methods[categories.index("compound")] == "tag"


def test_gansner_tse93_layer_assignment_by_tag():
    item = _item(
        id="gansner-tse93",
        title="A Technique for Drawing Directed Graphs",
        tags=["dot", "hierarchical", "layered", "rank"],
    )
    categories, methods = classify_item(item)
    assert "layer-assignment" in categories
    assert methods[categories.index("layer-assignment")] == "tag"


def test_left_edge_packing_by_tag():
    item = _item(
        title="Wire routing by optimizing channel assignment",
        tags=["packing", "channel-routing", "left-edge"],
    )
    categories, _ = classify_item(item)
    assert "packing" in categories
    assert "routing" in categories


def test_keyword_fallback_crossing():
    item = _item(
        title="Using Sifting for k-Layer Straightline Crossing Minimization",
        tags=["graph-drawing"],
        abstract="We study crossing minimization in layered graphs.",
    )
    categories, methods = classify_item(item)
    assert "crossing" in categories
    assert "layer-assignment" in categories
    assert methods[categories.index("crossing")] == "keyword"


def test_multi_category_compound_and_crossing():
    item = _item(
        title="Applying Crossing Reduction Strategies to Layered Compound Graphs",
        tags=["compound", "crossing", "layered"],
    )
    categories, _ = classify_item(item)
    assert "compound" in categories
    assert "crossing" in categories
    assert "layer-assignment" in categories


def test_uncategorized_generic_title():
    item = _item(
        title="R: A Language and Environment for Statistical Computing",
        tags=["openalex", "graph-drawing"],
    )
    categories, _ = classify_item(item)
    assert categories == []


def test_ports_maps_to_constraints_and_routing():
    assert "constraints" in categories_from_tags(["ports"])
    assert "routing" in categories_from_tags(["ports"])


def test_expanded_compaction_keywords():
    item = _item(
        title="Optimal Compaction of Orthogonal Grid Drawings",
        tags=["graph-drawing"],
        abstract="We minimize total edge length via two-dimensional compaction.",
    )
    categories, methods = classify_item(item)
    assert "compaction" in categories
    assert methods[categories.index("compaction")] == "keyword"


def test_expanded_packing_keywords():
    item = _item(
        title="Disconnected Graph Layout and the Polyomino Packing Approach",
        tags=["graph-drawing"],
        abstract="We pack disconnected components using a polyomino packing heuristic.",
    )
    categories, _ = classify_item(item)
    assert "packing" in categories


def test_expanded_overlap_keywords():
    item = _item(
        title="Efficient Node Overlap Removal Using a Proximity Stress Model",
        tags=["graph-drawing"],
        abstract="PRISM removes node overlap with a proximity stress model.",
    )
    categories, _ = classify_item(item)
    assert "overlap" in categories


def test_new_tag_maps():
    assert "coordinate-assignment" in categories_from_tags(["brandes-koepf"])
    assert "packing" in categories_from_tags(["disconnected"])
    assert "overlap" in categories_from_tags(["prism"])
    # orthogonal touches both routing and compaction
    cats = categories_from_tags(["orthogonal"])
    assert "routing" in cats and "compaction" in cats


# --- P1a graph-analytics expansion categories ---


def test_graph_analytics_tag_maps():
    assert "graph-analytics" in categories_from_tags(["pagerank"])
    assert "community-detection" in categories_from_tags(["louvain"])
    assert "community-detection" in categories_from_tags(["leiden"])
    assert "spectral-methods" in categories_from_tags(["spectral-clustering"])
    assert "graph-embeddings" in categories_from_tags(["node2vec"])
    assert "graph-embeddings" in categories_from_tags(["gnn"])
    assert "dynamic-graphs" in categories_from_tags(["temporal"])
    # edge bundling is its own edge-routing category (bundling ≠ pipeline routing)
    assert "edge-routing" in categories_from_tags(["edge-bundling"])
    # connector/orthogonal routing tags land in both edge-routing and routing
    cats = categories_from_tags(["orthogonal-routing"])
    assert "edge-routing" in cats and "routing" in cats


def test_community_detection_keyword_fallback():
    item = _item(
        title="Fast unfolding of communities in large networks",
        tags=["graph-drawing"],
        abstract="A modularity optimization heuristic (Louvain) for community detection.",
    )
    categories, methods = classify_item(item)
    assert "community-detection" in categories
    assert methods[categories.index("community-detection")] == "keyword"


def test_graph_analytics_and_embeddings_keyword_fallback():
    pagerank = _item(
        title="The PageRank Citation Ranking",
        tags=["graph-drawing"],
        abstract="Personalized pagerank as a link analysis centrality measure.",
    )
    cats, _ = classify_item(pagerank)
    assert "graph-analytics" in cats

    node2vec = _item(
        title="node2vec: Scalable Feature Learning for Networks",
        tags=["graph-drawing"],
        abstract="A node embedding / graph representation learning method.",
    )
    cats, _ = classify_item(node2vec)
    assert "graph-embeddings" in cats
