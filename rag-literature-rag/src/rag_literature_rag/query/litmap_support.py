"""Tool-side glue for `rag_common.litmap`: wires this tool's champion LanceDB index,
citation graph, and manifest into the shared, tool-agnostic litmap functions.

Kept separate from `cli.py` so both `rag-literature-rag` and `rag-literature-rag` can carry
an (almost) identical copy — the two tools mirror each other's `ingest/index.py`,
`paths.py`, `query/citation_rank.py`, `query/search.py`, and `manifest.py` shapes closely
enough that this module only ever imports through those, never anything GLR-specific.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_common.litmap import doc_vectors as dv
from rag_common.litmap.types import CitationAdjacency
from rag_literature_rag.manifest import load_manifest
from rag_literature_rag.paths import ProfileIndexPaths, profile_index_paths
from rag_literature_rag.query.citation_rank import CitationGraph, load_graph_cached


def doc_meta_map() -> dict[str, dict[str, Any]]:
    """{doc_id: {"title": str, "tags": list[str]}} for cluster labeling."""
    manifest = load_manifest()
    return {item.id: {"title": item.title, "tags": item.tags} for item in manifest.items}


def _index_fingerprint(lance_dir: Path) -> str:
    """Cheap change-detector for the doc-vector cache: latest mtime under the LanceDB
    directory. Any re-ingest touches at least one file, invalidating the cache."""
    if not lance_dir.exists():
        return "missing"
    mtimes = [p.stat().st_mtime for p in lance_dir.rglob("*") if p.is_file()]
    return f"{max(mtimes):.3f}" if mtimes else "empty"


def litmap_dir(paths: ProfileIndexPaths) -> Path:
    return paths.root / "litmap"


def load_or_build_doc_vectors(
    profile: str | None = None,
    *,
    rebuild: bool = False,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[dict[str, list[float]], ProfileIndexPaths]:
    """Load the cached litmap doc-vector store for `profile`, rebuilding it from the
    champion LanceDB chunk index if missing or stale (profile/index-fingerprint mismatch).

    Raises `FileNotFoundError` (caller renders the "no index" message) if the profile has
    no LanceDB index at all — mirrors the existing error style used by `embed indexes`.
    """
    paths = profile_index_paths(profile)
    if not paths.lance_dir.exists():
        raise FileNotFoundError(
            f"No index for profile {paths.profile!r} at {paths.lance_dir}. "
            f"Run: rag-literature-rag ingest --profile {paths.profile}"
        )
    fingerprint = _index_fingerprint(paths.lance_dir)
    out_dir = litmap_dir(paths)
    if not rebuild:
        cached = dv.load_doc_vectors(
            out_dir, embed_profile=paths.profile, index_fingerprint=fingerprint
        )
        if cached is not None:
            return cached, paths
    vectors = dv.build_doc_vectors_from_lancedb(paths.lance_dir, progress_cb=progress_cb)
    dv.save_doc_vectors(
        out_dir, vectors, embed_profile=paths.profile, index_fingerprint=fingerprint
    )
    return vectors, paths


def citation_graph() -> CitationGraph | None:
    return load_graph_cached()


def citation_adjacency(graph: CitationGraph | None = None) -> CitationAdjacency | None:
    g = graph if graph is not None else citation_graph()
    if g is None:
        return None
    return CitationAdjacency.from_graph(g)


def load_years(graph: CitationGraph) -> dict[str, int]:
    """`{label: year}` for every paper the citation store has a year for, keyed the same
    way lineage/main-path edges are (corpus doc_id where the graph can resolve one,
    otherwise the raw OpenAlex id) — so it lines up directly with edges built from
    `graph.oa_to_doc.get(oa, oa)` labels."""
    from rag_literature_rag import citation_store as cs

    if not cs.CITATIONS_DB_PATH.exists():
        return {}
    db = cs.connect()
    try:
        rows = db.execute("SELECT oa_id, year FROM papers WHERE year IS NOT NULL")
        return {graph.oa_to_doc.get(oa, oa): year for oa, year in rows}
    finally:
        db.close()
