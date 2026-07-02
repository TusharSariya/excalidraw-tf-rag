"""Tool-side glue for `rag_common.litmap`: wires this tool's champion LanceDB index,
citation graph, and manifest into the shared, tool-agnostic litmap functions.

Kept separate from `cli.py` so both `rag-literature-rag` and `rag-literature-rag` can carry
an (almost) identical copy — the two tools mirror each other's `ingest/index.py`,
`paths.py`, `query/citation_rank.py`, `query/search.py`, and `manifest.py` shapes closely
enough that this module only ever imports through those, never anything GLR-specific.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_common.litmap import doc_vectors as dv
from rag_common.litmap.types import CitationAdjacency
from rag_literature_rag.manifest import load_manifest
from rag_literature_rag.paths import ProfileIndexPaths, list_profile_indexes, profile_index_paths
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


def _resolve_paths_with_fallback(profile: str | None) -> ProfileIndexPaths:
    """`profile_index_paths`, but with the default-profile footgun closed: if `profile`
    wasn't given explicitly and the *default* profile (env/`.env`-resolved) has no built
    index, and exactly one *other* profile index is built, warn and use that one instead
    of raising. An explicit `profile` argument is never silently overridden — a user who
    names a profile that has no index should still see the current clear error. Ambiguous
    cases (zero or >1 other built indexes) also fall through to the current error, now
    listing the built profiles when there's more than one so the fix (`--profile ...`) is
    obvious."""
    paths = profile_index_paths(profile)
    if paths.lance_dir.exists():
        return paths
    if profile is not None:
        raise FileNotFoundError(
            f"No index for profile {paths.profile!r} at {paths.lance_dir}. "
            f"Run: rag-literature-rag ingest --profile {paths.profile}"
        )

    built = [p for p in list_profile_indexes() if p.lance_dir.exists()]
    if len(built) == 1:
        fallback = built[0]
        print(
            f"warning: default profile {paths.profile!r} has no index at {paths.lance_dir}; "
            f"falling back to the only built profile index, {fallback.profile!r} "
            f"(pass --profile to silence this).",
            file=sys.stderr,
        )
        return fallback
    if len(built) > 1:
        names = ", ".join(repr(p.profile) for p in built)
        raise FileNotFoundError(
            f"No index for default profile {paths.profile!r} at {paths.lance_dir}, and "
            f"{len(built)} other profile indexes are built ({names}) — ambiguous, pass "
            f"--profile to pick one."
        )
    raise FileNotFoundError(
        f"No index for profile {paths.profile!r} at {paths.lance_dir}. "
        f"Run: rag-literature-rag ingest --profile {paths.profile}"
    )


def load_or_build_doc_vectors(
    profile: str | None = None,
    *,
    rebuild: bool = False,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[dict[str, list[float]], ProfileIndexPaths]:
    """Load the cached litmap doc-vector store for `profile`, rebuilding it from the
    champion LanceDB chunk index if missing or stale (profile/index-fingerprint mismatch).

    Raises `FileNotFoundError` (caller renders the "no index" message) if the profile has
    no LanceDB index at all, *and* no unambiguous fallback exists — see
    `_resolve_paths_with_fallback`. Mirrors the existing error style used by `embed
    indexes`.
    """
    paths = _resolve_paths_with_fallback(profile)
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
