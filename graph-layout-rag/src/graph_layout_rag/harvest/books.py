from __future__ import annotations

from graph_layout_rag.manifest import ManifestItem


def book_metadata_stubs() -> list[ManifestItem]:
    return [
        ManifestItem(
            id="book-dett99",
            title="Graph Drawing: Algorithms for the Visualization of Graphs",
            authors=["Di Battista", "Eades", "Tamassia", "Tollis"],
            year=1999,
            source="book",
            url="https://doi.org/10.1007/978-1-4612-0353-7",
            contentType="text/metadata",
            status="metadata_only",
            tags=["book", "survey"],
            doi="10.1007/978-1-4612-0353-7",
            abstract="Classic graph drawing textbook (Prentice Hall). Use handbook chapter PDFs for open content.",
        ),
        ManifestItem(
            id="book-kaufmann-wagner",
            title="Drawing Graphs: Methods and Models",
            authors=["Kaufmann", "Wagner"],
            year=2001,
            source="book",
            url="https://doi.org/10.1007/3-540-44969-8",
            contentType="text/metadata",
            status="metadata_only",
            tags=["book"],
            doi="10.1007/3-540-44969-8",
            abstract="Springer LNCS survey volume; typically paywalled.",
        ),
        ManifestItem(
            id="book-junger-mutzel",
            title="Graph Drawing Software",
            authors=["Jünger", "Mutzel"],
            year=2002,
            source="book",
            url="https://doi.org/10.1007/978-3-642-55946-3",
            contentType="text/metadata",
            status="metadata_only",
            tags=["book", "software"],
            doi="10.1007/978-3-642-55946-3",
            abstract="Graph drawing software systems including Graphviz-era tools.",
        ),
        ManifestItem(
            id="book-tamassia-handbook",
            title="Handbook of Graph Drawing and Visualization",
            authors=["Tamassia"],
            year=2013,
            source="book",
            url="https://doi.org/10.1201/b15385",
            contentType="text/metadata",
            status="metadata_only",
            tags=["book", "survey", "handbook"],
            doi="10.1201/b15385",
            abstract=(
                "Comprehensive reference edited by Tamassia covering all major graph drawing "
                "topics: layered, force-directed, orthogonal, planar, constrained. Many chapters "
                "available as open PDFs from authors' sites. CRC Press 2013."
            ),
        ),
        ManifestItem(
            id="book-nishizeki-rahman",
            title="Planar Graph Drawing",
            authors=["Nishizeki", "Rahman"],
            year=2004,
            source="book",
            url="https://doi.org/10.1142/5648",
            contentType="text/metadata",
            status="metadata_only",
            tags=["book", "planar"],
            doi="10.1142/5648",
            abstract=(
                "Textbook on planar graph drawing algorithms: visibility representations, "
                "straight-line drawings, convex drawings, box-rectangular drawings. "
                "World Scientific 2004."
            ),
        ),
        ManifestItem(
            id="book-healy-nikolov-hierarchical",
            title="Hierarchical Drawing Algorithms",
            authors=["Healy", "Nikolov"],
            year=2013,
            source="book",
            url="https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/hierarchical.pdf",
            contentType="application/pdf",
            status="metadata_only",
            tags=["book", "hierarchical", "layered", "sugiyama", "survey"],
            abstract=(
                "Survey chapter on hierarchical/Sugiyama-framework drawing algorithms. "
                "Covers layer assignment, crossing minimization, coordinate assignment. "
                "From the Handbook of Graph Drawing and Visualization."
            ),
        ),
        ManifestItem(
            id="book-kobourov-force-directed",
            title="Force-Directed Drawing Algorithms",
            authors=["Kobourov"],
            year=2013,
            source="book",
            url="https://cs.brown.edu/people/rtamassia/gdhandbook/chapters/force-directed.pdf",
            contentType="application/pdf",
            status="metadata_only",
            tags=["book", "force-directed", "survey"],
            abstract=(
                "Survey chapter on force-directed graph drawing: spring-embedder, stress "
                "majorization, multilevel methods, scalability. From the Handbook of Graph "
                "Drawing and Visualization."
            ),
        ),
    ]
