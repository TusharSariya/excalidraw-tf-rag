"""Pipeline-layout taxonomy for PDF catalog classification."""

from __future__ import annotations

PIPELINE_CATEGORIES: tuple[str, ...] = (
    "layer-assignment",
    "crossing",
    "compound",
    "constraints",
    "coordinate-assignment",
    "routing",
    "compaction",
    "packing",
    "overlap",
    # --- general graph-theory / graph-analytics categories (P1a expansion) ---
    # The corpus grew beyond the Sugiyama pipeline to cover graph analytics and
    # representation learning. These six are self-standing subject areas, not
    # pipeline phases; they are appended (never inserted) so the pipeline order
    # above stays stable for anything that indexes by position.
    "graph-analytics",  # PageRank / centrality / HITS / random-walk ranking
    "community-detection",  # Louvain / Leiden / modularity / graph partitioning
    "spectral-methods",  # spectral clustering / graph Laplacian / eigenvectors
    "graph-embeddings",  # node2vec / DeepWalk / LINE / GNN-based layout
    "edge-routing",  # edge bundling + standalone orthogonal connector routing
    "dynamic-graphs",  # dynamic / streaming / temporal graph layout
)

UNCATEGORIZED = "uncategorized"

# Manifest harvest tags → pipeline categories (many-to-many).
TAG_TO_CATEGORIES: dict[str, list[str]] = {
    # layer-assignment
    "layer-assignment": ["layer-assignment"],
    "layered": ["layer-assignment"],
    "sugiyama": ["layer-assignment"],
    "hierarchical": ["layer-assignment"],
    "dagre": ["layer-assignment"],
    "minimum-width": ["layer-assignment"],
    "coffman-graham": ["layer-assignment"],
    "wrapping": ["layer-assignment"],
    "rank": ["layer-assignment"],
    "dot": ["layer-assignment"],
    "elk-kieler": ["layer-assignment"],
    "elk": ["layer-assignment"],
    # crossing
    "crossing": ["crossing"],
    "sifting": ["crossing"],
    # compound
    "compound": ["compound"],
    "grouped": ["compound"],
    "clustering": ["compound"],
    "nested": ["compound"],
    # constraints
    "constraints": ["constraints"],
    "elastic": ["constraints"],
    "vpsc": ["constraints"],
    "ports": ["constraints", "routing"],
    # coordinate-assignment
    "coordinate-assignment": ["coordinate-assignment"],
    "brandes-koepf": ["coordinate-assignment"],
    "tidy-tree": ["coordinate-assignment"],
    # routing
    "routing": ["routing"],
    "channel-routing": ["routing"],
    "orthogonal": ["routing", "compaction"],
    # compaction
    "compaction": ["compaction"],
    "compact": ["compaction"],
    "scanline": ["compaction"],
    # packing
    "packing": ["packing"],
    "left-edge": ["packing"],
    "strip-packing": ["packing"],
    "floorplanning": ["packing"],
    "sequence-pair": ["packing"],
    "b-star-tree": ["packing"],
    "disconnected": ["packing"],
    # overlap
    "overlap": ["overlap"],
    "mental-map": ["overlap"],
    "clutter": ["overlap"],
    "prism": ["overlap"],
    "node-overlap-removal": ["overlap"],
    # graph-analytics
    "graph-analytics": ["graph-analytics"],
    "pagerank": ["graph-analytics"],
    "centrality": ["graph-analytics"],
    "hits": ["graph-analytics"],
    "random-walk": ["graph-analytics"],
    "link-analysis": ["graph-analytics"],
    # community-detection
    "community-detection": ["community-detection"],
    "louvain": ["community-detection"],
    "leiden": ["community-detection"],
    "modularity": ["community-detection"],
    "graph-partitioning": ["community-detection"],
    "girvan-newman": ["community-detection"],
    # spectral-methods
    "spectral": ["spectral-methods"],
    "spectral-clustering": ["spectral-methods"],
    "laplacian": ["spectral-methods"],
    "normalized-cuts": ["spectral-methods"],
    "eigenvector": ["spectral-methods"],
    # graph-embeddings
    "graph-embedding": ["graph-embeddings"],
    "node-embedding": ["graph-embeddings"],
    "node2vec": ["graph-embeddings"],
    "deepwalk": ["graph-embeddings"],
    "line-embedding": ["graph-embeddings"],
    "graph-neural-network": ["graph-embeddings"],
    "gnn": ["graph-embeddings"],
    "representation-learning": ["graph-embeddings"],
    # edge-routing (edge bundling + standalone connector routing). "orthogonal"
    # already maps to routing+compaction above; the bundling tags are new and
    # land here rather than in the pipeline `routing` phase.
    "edge-bundling": ["edge-routing"],
    "orthogonal-routing": ["edge-routing", "routing"],
    "connector-routing": ["edge-routing", "routing"],
    "libavoid": ["edge-routing", "routing"],
    # dynamic-graphs
    "dynamic": ["dynamic-graphs"],
    "dynamic-graph": ["dynamic-graphs"],
    "temporal": ["dynamic-graphs"],
    "streaming": ["dynamic-graphs"],
    "storyline": ["dynamic-graphs"],
}

# Title/abstract phrase lists per pipeline category (keyword fallback).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "layer-assignment": [
        "layer assignment",
        "graph layering",
        "layered drawing",
        "layered graph",
        "minimum width layering",
        "minimum-width layering",
        "node promotion",
        "rank assignment",
        "network simplex",
        "coffman graham",
        "coffman-graham",
        "as-late-as-possible",
        "alap scheduling",
        "layer reassignment",
        "sugiyama",
        "hierarchical layout",
        "hierarchical drawing",
        "global ranking",
        "dummy node",
        "dag layout",
    ],
    "crossing": [
        "crossing minimization",
        "crossing reduction",
        "edge crossing",
        "two-layer crossing",
        "k-layer crossing",
        "sifting",
        "straightline crossing",
        "crossing number",
    ],
    "compound": [
        "compound graph",
        "compound digraph",
        "compound directed",
        "nested graph",
        "cluster border",
        "clustered graph",
        "grouped network",
        "hierarchical cluster",
    ],
    "constraints": [
        "separation constraint",
        "vpsc",
        "variable placement",
        "ipsep-cola",
        "dig-cola",
        "constrained layout",
        "port constraint",
        "containment constraint",
        "quadratic program",
    ],
    "coordinate-assignment": [
        "coordinate assignment",
        "horizontal coordinate",
        "vertical coordinate",
        "brandes köpf",
        "brandes-köpf",
        "brandes and köpf",
        "x coordinate assignment",
        "x-coordinate assignment",
        "node positioning within layer",
        "median heuristic",
        "priority method",
        "tidy tree",
        "tidier drawings of trees",
        "horizontal compaction",
    ],
    "routing": [
        "edge routing",
        "orthogonal routing",
        "connector routing",
        "port routing",
        "orthogonal connector",
        "wire routing",
        "channel assignment",
        "track assignment",
    ],
    "compaction": [
        "compaction",
        "one-dimensional compaction",
        "1d compaction",
        "two-dimensional compaction",
        "2d compaction",
        "orthogonal compaction",
        "longest-path compaction",
        "longest path compaction",
        "pushback",
        "symbolic layout compaction",
        "constraint graph compaction",
        "dual graph compaction",
        "scanline compaction",
        "shadow propagation",
        "minimize total edge length",
    ],
    "packing": [
        "strip packing",
        "skyline heuristic",
        "rectangular packing",
        "rectangle packing",
        "bin packing",
        "guillotine",
        "left edge algorithm",
        "left-edge algorithm",
        "bottom-left",
        "sequence-pair",
        "b*-tree",
        "b-star tree",
        "floorplan packing",
        "polyomino packing",
        "component packing",
        "disconnected components",
        "disconnected graph layout",
    ],
    "overlap": [
        "overlap removal",
        "node overlap",
        "node overlap removal",
        "remove node overlap",
        "mental map",
        "layout adjustment",
        "cluster busting",
        "clutter reduction",
        "proximity stress",
        "prism",
        "force scan",
        "growing a tree",
    ],
    "graph-analytics": [
        "pagerank",
        "page rank",
        "personalized pagerank",
        "personalised pagerank",
        "hits algorithm",
        "hubs and authorities",
        "authoritative sources",
        "link analysis",
        "centrality measure",
        "betweenness centrality",
        "eigenvector centrality",
        "closeness centrality",
        "random walk ranking",
    ],
    "community-detection": [
        "community detection",
        "community structure",
        "modularity optimization",
        "modularity optimisation",
        "modularity maximization",
        "louvain",
        "leiden",
        "graph partitioning",
        "network partitioning",
        "girvan-newman",
        "girvan newman",
        "label propagation",
    ],
    "spectral-methods": [
        "spectral clustering",
        "spectral partitioning",
        "graph laplacian",
        "normalized cut",
        "normalized cuts",
        "normalised cuts",
        "eigenvector method",
        "fiedler vector",
        "laplacian eigenmap",
    ],
    "graph-embeddings": [
        "node embedding",
        "node2vec",
        "deepwalk",
        "graph embedding",
        "network embedding",
        "graph representation learning",
        "representation learning on graphs",
        "graph neural network",
        "graph convolutional network",
        "graph attention network",
        "graph autoencoder",
        "graph drawing with neural network",
        "deep learning based graph drawing",
    ],
    "edge-routing": [
        "edge bundling",
        "edge bundles",
        "hierarchical edge bundling",
        "force-directed edge bundling",
        "orthogonal connector routing",
        "orthogonal edge routing",
        "connector routing",
        "hyperedge routing",
    ],
    "dynamic-graphs": [
        "dynamic graph drawing",
        "dynamic graph visualization",
        "dynamic graph visualisation",
        "dynamic network visualization",
        "temporal graph",
        "temporal network",
        "streaming graph",
        "evolving graph",
        "graph animation",
        "storyline layout",
    ],
}


def categories_from_tags(tags: list[str]) -> set[str]:
    out: set[str] = set()
    for tag in tags:
        out.update(TAG_TO_CATEGORIES.get(tag, []))
    return out


def categories_from_keywords(text: str) -> set[str]:
    hay = text.lower()
    out: set[str] = set()
    for category, phrases in CATEGORY_KEYWORDS.items():
        if any(phrase in hay for phrase in phrases):
            out.add(category)
    return out
