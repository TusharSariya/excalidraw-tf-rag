"""Curated seed papers discovered via maintainer/founder social mining.

Sources: Graphviz authors (Gansner, North, Ellson), Terrastruct/D2-TALA team,
ELK team GitHub discussions, HN/Reddit threads, yWorks technical blog posts.

Each entry follows the same shape as curated.py. Papers with open PDFs are
listed under MAINTAINER_PDFS; metadata-only (paywalled/no OA copy) go under
MAINTAINER_METADATA.
"""

from __future__ import annotations

from graph_layout_rag.harvest.curated import _harvest_curated_pdf
from graph_layout_rag.harvest.parallel import parallel_map
from graph_layout_rag.manifest import ManifestItem

MAINTAINER_PDFS: list[dict] = [
    {
        "id": "gansner-north-improved-spring-1996",
        "title": "An Improved Spring Embedder",
        "authors": ["Gansner", "Koren", "North"],
        "year": 2004,
        "url": "https://www.graphviz.org/Documentation/GKN04.pdf",
        "doi": "10.1007/978-3-540-31843-9_2",
        "source": "graphviz.org",
        "tags": ["force-directed", "spring", "stress", "graphviz", "maintainer-seed"],
    },
    {
        "id": "gansner-koren-north-topological-fisheye-2005",
        "title": "Topological Fisheye Views",
        "authors": ["Gansner", "Koren", "North"],
        "year": 2005,
        "url": "https://www.graphviz.org/Documentation/GKN05.pdf",
        "doi": "10.1109/infvis.2005.1532128",
        "source": "graphviz.org",
        "tags": ["fisheye", "focus+context", "graphviz", "maintainer-seed"],
    },
    {
        "id": "gansner-north-graph-drawing-2000",
        "title": "An Open Graph Visualization System and Its Applications to Software Engineering",
        "authors": ["Gansner", "North"],
        "year": 2000,
        "url": "https://www.graphviz.org/Documentation/GN99.pdf",
        "doi": "10.1002/1097-024x(200009)30:11<1203::aid-spe338>3.0.co;2-n",
        "source": "graphviz.org",
        "tags": ["graphviz", "software-engineering", "overview", "maintainer-seed"],
    },
    {
        "id": "koutsofios-north-drawing-graphs-1996",
        "title": "Drawing Graphs with dot",
        "authors": ["Koutsofios", "North"],
        "year": 1996,
        "url": "https://www.graphviz.org/pdf/dotguide.pdf",
        "source": "graphviz.org",
        "tags": ["graphviz", "dot", "user-guide", "maintainer-seed"],
    },
    {
        "id": "ellson-graphviz-gd2001",
        "title": "Graphviz — Open Source Graph Drawing Tools",
        "authors": ["Ellson", "Gansner", "Koutsofios", "North", "Woodhull"],
        "year": 2001,
        "url": "https://www.graphviz.org/Documentation/EGKNW03.pdf",
        "doi": "10.1007/3-540-45848-4_57",
        "source": "graphviz.org",
        "tags": ["graphviz", "open-source", "tools", "gd2001", "maintainer-seed"],
    },
    {
        "id": "hu-efficient-force-directed-2005",
        "title": "Efficient and High Quality Force-Directed Graph Drawing",
        "authors": ["Hu"],
        "year": 2005,
        "url": "https://www.mathematica-journal.com/2006/09/19/graph-drawing/",
        "doi": "10.3888/tmj.10.1-2.37",
        "source": "graphviz.org",
        "tags": ["force-directed", "multilevel", "sfdp", "graphviz", "maintainer-seed"],
    },
]

MAINTAINER_METADATA: list[dict] = [
    {
        "id": "gansner-koren-drawing-graphs-spectral-2004",
        "title": "Graph Drawing by Stress Majorization",
        "authors": ["Gansner", "Koren", "North"],
        "year": 2004,
        "url": "https://doi.org/10.1007/978-3-540-31843-9_25",
        "doi": "10.1007/978-3-540-31843-9_25",
        "source": "springer",
        "tags": ["stress", "majorization", "force-directed", "neato", "graphviz", "maintainer-seed"],
        "abstract": (
            "Reformulates graph layout as a stress minimization problem and solves it via "
            "stress majorization, converging faster than gradient descent. Used in Graphviz "
            "neato mode. One of the foundational algorithms in force-directed graph drawing."
        ),
    },
    {
        "id": "koren-drawing-graphs-spectral-2003",
        "title": "On Spectral Graph Drawing",
        "authors": ["Koren"],
        "year": 2003,
        "url": "https://doi.org/10.1007/3-540-45071-8_47",
        "doi": "10.1007/3-540-45071-8_47",
        "source": "springer",
        "tags": ["spectral", "eigenvector", "force-directed", "graphviz", "maintainer-seed"],
        "abstract": (
            "Derives graph drawing algorithms from the spectral decomposition of the graph "
            "Laplacian, producing aesthetically pleasing drawings by optimizing global "
            "symmetry via eigenvectors. Influences sfdp/neato in Graphviz."
        ),
    },
    {
        "id": "north-incremental-layout-1996",
        "title": "Incremental Layout in DynaDAG",
        "authors": ["North", "Woodhull"],
        "year": 2001,
        "url": "https://doi.org/10.1007/3-540-45848-4_15",
        "doi": "10.1007/3-540-45848-4_15",
        "source": "springer",
        "tags": ["incremental", "dynamic", "dag", "graphviz", "maintainer-seed"],
        "abstract": (
            "DynaDAG: an incremental layout algorithm for dynamic directed acyclic graphs. "
            "Maintains a stable layout as nodes/edges are added or removed, building on "
            "the Sugiyama framework. Foundational for live/streaming graph visualization."
        ),
    },
    {
        "id": "terrastruct-blog-layout-engines-2023",
        "title": "Diagram layout engines: A look under the hood",
        "authors": ["Batista"],
        "year": 2023,
        "url": "https://www.terrastruct.com/blog/post/layout-engines/",
        "source": "terrastruct",
        "tags": ["blog", "layout-engines", "overview", "d2", "tala", "maintainer-seed"],
        "abstract": (
            "Terrastruct blog overview of diagram layout engines (ELK, dagre, D2/TALA). "
            "Discusses trade-offs between layered, force-directed, and orthogonal approaches "
            "from the perspective of D2's authors. References canonical papers for each approach."
        ),
    },
    {
        "id": "wybrow-marriott-stuckey-orthogonal-connectors-2010",
        "title": "Orthogonal Connector Routing",
        "authors": ["Wybrow", "Marriott", "Stuckey"],
        "year": 2010,
        "url": "https://doi.org/10.1007/978-3-642-11805-0_37",
        "doi": "10.1007/978-3-642-11805-0_37",
        "source": "springer",
        "tags": ["routing", "orthogonal", "connector", "libavoid", "yworks", "maintainer-seed"],
        "abstract": (
            "Libavoid-style orthogonal connector routing: finds short, non-overlapping "
            "rectilinear paths between nodes with port constraints. Used in Inkscape, "
            "yEd, and other diagram editors. Key algorithm for diagram layout routing."
        ),
    },
    {
        "id": "wybrow-marriott-stuckey-incremental-routing-2008",
        "title": "Incremental Connector Routing",
        "authors": ["Wybrow", "Marriott", "Stuckey"],
        "year": 2008,
        "url": "https://doi.org/10.1007/978-3-540-77537-9_15",
        "doi": "10.1007/978-3-540-77537-9_15",
        "source": "springer",
        "tags": ["routing", "incremental", "connector", "libavoid", "maintainer-seed"],
        "abstract": (
            "Extends Libavoid to support incremental re-routing as diagram elements move "
            "without recomputing from scratch. Critical for interactive diagram editing "
            "where connectors must update smoothly as nodes are dragged."
        ),
    },
    {
        "id": "dobkin-gurari-tal-visibility-representation-1997",
        "title": "The Complexity of Generating Optimal Visibility Representations",
        "authors": ["Dobkin", "Gurari", "Tel"],
        "year": 1997,
        "url": "https://doi.org/10.1007/BF02122698",
        "doi": "10.1007/bf02122698",
        "source": "springer",
        "tags": ["visibility", "planar", "complexity", "maintainer-seed"],
        "abstract": (
            "Studies the complexity of computing optimal visibility representations of "
            "planar graphs, where vertices map to horizontal segments and edges to vertical "
            "segments. Foundational for orthogonal and visibility-based layout algorithms."
        ),
    },
]


def harvest_maintainer_seeds(*, dry_run: bool = False, workers: int | None = None) -> list[ManifestItem]:
    """Return curated seeds from maintainer/founder social mining."""
    results: list[ManifestItem] = []

    results.extend(
        parallel_map(
            lambda spec: _harvest_curated_pdf(spec, dry_run=dry_run),
            MAINTAINER_PDFS,
            workers=workers,
        )
    )

    for spec in MAINTAINER_METADATA:
        results.append(
            ManifestItem(
                id=spec["id"],
                title=spec["title"],
                authors=spec.get("authors", []),
                year=spec.get("year"),
                source=spec.get("source", "curated"),
                url=spec["url"],
                contentType="text/metadata",
                status="metadata_only",
                tags=[*spec.get("tags", []), "maintainer-seed"],
                doi=spec.get("doi"),
                abstract=spec.get("abstract"),
            )
        )

    return results
