"""`map` CLI group — the knowledge-map layer (E1) over this corpus.

Thin click wiring around `rag_common.litmap`: clusters (Leiden over the fused
text+citation kNN graph), lineage (main-path analysis over a topic subgraph), similar
(Connected-Papers-style neighborhood), and seminal (PPR anchor discovery — a *view* over
the citation graph, never fed back into retrieval ranking: that's a settled NULL).

Mirrored in rag-literature-rag — when fixing something here, check the twin.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

_NO_CITE_STORE = "No citation store yet. Run: graph-layout-rag cite enrich"


@click.group("map")
def map_group() -> None:
    """Knowledge maps: clusters, lineage, similar papers, seminal anchors."""


def _titles() -> dict[str, str]:
    from graph_layout_rag.query.litmap_support import doc_meta_map

    return {doc_id: meta.get("title") or doc_id for doc_id, meta in doc_meta_map().items()}


def _require_graph() -> Any:
    from graph_layout_rag.query.litmap_support import citation_graph

    graph = citation_graph()
    if graph is None:
        click.echo(_NO_CITE_STORE, err=True)
        sys.exit(1)
    return graph


def _seed_doc_ids(query_or_doc_id: str, *, top: int = 10, profile: str | None = None) -> list[str]:
    """A doc_id resolves to itself; anything else runs the tool's existing search."""
    from graph_layout_rag.query.litmap_support import doc_meta_map
    from graph_layout_rag.query.search import search

    if query_or_doc_id in doc_meta_map():
        return [query_or_doc_id]
    results = search(query_or_doc_id, top=top, embed_profile=profile)
    return [r["doc_id"] for r in results if r.get("doc_id")]


@map_group.command("clusters")
@click.option("--profile", default=None, help="Embed profile whose index supplies doc vectors.")
@click.option(
    "--alpha",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Citation-signal weight in the rank fusion (0 = pure text, 1 = pure citation).",
)
@click.option(
    "--llm-labels", is_flag=True, help="Label clusters with the local LLM (extractive fallback)."
)
@click.option(
    "--top-k", default=10, show_default=True, help="Neighbors per doc in the fused kNN graph."
)
@click.option("--seed", default=42, show_default=True, help="Leiden random seed.")
@click.option("--rebuild-vectors", is_flag=True, help="Force a doc-vector rebuild from the index.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def map_clusters_cmd(
    profile: str | None,
    alpha: float,
    llm_labels: bool,
    top_k: int,
    seed: int,
    rebuild_vectors: bool,
    as_json: bool,
) -> None:
    """Cluster the corpus into topical communities (Leiden over fused text+citation kNN)."""
    from graph_layout_rag.query.litmap_support import (
        citation_adjacency,
        doc_meta_map,
        load_or_build_doc_vectors,
    )
    from rag_common.litmap.communities import build_clusters
    from rag_common.litmap.similarity import bib_coupling_all, cosine_knn_all, fuse_neighbors

    def _progress(n_chunks: int) -> None:
        click.echo(f"map clusters: pooled {n_chunks} chunk vectors...", err=True)

    try:
        vectors, paths = load_or_build_doc_vectors(
            profile, rebuild=rebuild_vectors, progress_cb=_progress
        )
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    if not vectors:
        click.echo(f"Index for profile {paths.profile!r} has no chunks.", err=True)
        sys.exit(1)

    adj = citation_adjacency()
    if adj is None and alpha > 0.0:
        click.echo(_NO_CITE_STORE, err=True)
        sys.exit(1)

    doc_ids = list(vectors)
    text_knn = cosine_knn_all(vectors, top_k=top_k)
    cite_knn = (
        bib_coupling_all(doc_ids, adj, top_k=top_k) if adj is not None and alpha > 0.0 else {}
    )

    edge_scores: dict[tuple[str, str], float] = {}
    for doc_id in doc_ids:
        fused = fuse_neighbors(
            text_knn.get(doc_id, []), cite_knn.get(doc_id, []), alpha=alpha, top_k=top_k
        )
        for n in fused:
            key = (doc_id, n.doc_id) if doc_id < n.doc_id else (n.doc_id, doc_id)
            edge_scores[key] = max(edge_scores.get(key, 0.0), n.score)
    edges = [(u, v, w) for (u, v), w in edge_scores.items()]

    result = build_clusters(
        edges,
        doc_meta_map(),
        all_doc_ids=doc_ids,
        seed=seed,
        use_llm=llm_labels,
    )
    result["profile"] = paths.profile
    result["alpha"] = alpha

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    titles = _titles()
    click.echo(
        f"{result['cluster_count']} clusters over {result['doc_count']} docs "
        f"(profile={paths.profile}, alpha={alpha})\n"
    )
    for cluster in result["clusters"]:
        click.echo(
            f"[{cluster['id']}] {cluster['label']}  (n={cluster['size']}, {cluster['label_method']})"
        )
        for doc_id in cluster["doc_ids"][:3]:
            click.echo(f"    - {titles.get(doc_id, doc_id)}")
        if cluster["size"] > 3:
            click.echo(f"    ... {cluster['size'] - 3} more")


@map_group.command("lineage")
@click.argument("query_or_doc_id")
@click.option("--profile", default=None, help="Embed profile used to search for topic seeds.")
@click.option("--max-nodes", default=500, show_default=True, help="Topic-subgraph size cap.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def map_lineage_cmd(
    query_or_doc_id: str, profile: str | None, max_nodes: int, as_json: bool
) -> None:
    """Trace the main intellectual lineage (SPC main path) of a topic or paper."""
    from graph_layout_rag.query.litmap_support import load_years
    from rag_common.litmap.mainpath import build_main_path

    graph = _require_graph()
    seeds = _seed_doc_ids(query_or_doc_id, profile=profile)
    if not seeds:
        click.echo(f"No documents match {query_or_doc_id!r}.", err=True)
        sys.exit(1)
    seed_oas = {graph.doc_to_oa[d] for d in seeds if d in graph.doc_to_oa}
    if not seed_oas:
        click.echo(
            f"None of the {len(seeds)} matching doc(s) are keyed into the citation graph. "
            "Run: graph-layout-rag cite enrich",
            err=True,
        )
        sys.exit(1)

    nodes = graph.neighbors_within(seed_oas, radius=1, max_nodes=max_nodes)

    def _label(oa: str) -> str:
        return graph.oa_to_doc.get(oa, oa)

    citation_edges = [
        (_label(src), _label(dst))
        for src in nodes
        for dst in graph.out_adj.get(src, ())
        if dst in nodes
    ]
    result = build_main_path(citation_edges, years=load_years(graph))
    result["query"] = query_or_doc_id
    result["seeds"] = seeds
    titles = _titles()
    result["path_titles"] = [titles.get(n, n) for n in result["main_path"]]

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    for warning in result["warnings"]:
        click.echo(f"warning: {warning}", err=True)
    if not result["main_path"]:
        click.echo(
            f"No citation edges among the {len(nodes)} papers around {query_or_doc_id!r} — "
            "no lineage to trace."
        )
        return
    click.echo(
        f"Main path for {query_or_doc_id!r} "
        f"({result['node_count']} papers, {result['edge_count']} edges considered):\n"
    )
    for i, node in enumerate(result["main_path"]):
        spc = f"  [SPC {result['edges'][i - 1]['spc']}]" if i else ""
        click.echo(f"  {'-> ' if i else '   '}{titles.get(node, node)}{spc}")


@map_group.command("similar")
@click.argument("doc_id")
@click.option("--top", default=10, show_default=True, help="Neighbors to return.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--dot", "as_dot", is_flag=True, help="Emit a Graphviz DOT graph.")
def map_similar_cmd(doc_id: str, top: int, as_json: bool, as_dot: bool) -> None:
    """Connected-Papers-style neighborhood: co-citation + coupling around one paper."""
    from graph_layout_rag.query.litmap_support import citation_adjacency, doc_meta_map
    from rag_common.litmap.neighborhood import neighborhood_graph, to_dot

    meta = doc_meta_map()
    if doc_id not in meta:
        click.echo(f"Unknown doc id: {doc_id!r} (not in the corpus manifest).", err=True)
        sys.exit(1)
    graph = _require_graph()
    adj = citation_adjacency(graph)

    candidates = [d for d in graph.doc_to_oa if d in meta]
    result = neighborhood_graph(doc_id, candidates, adj, top_n=top)
    titles = _titles()
    result["titles"] = {n: titles.get(n, n) for n in result["nodes"]}

    if as_dot:
        click.echo(to_dot(result))
        return
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"Papers similar to: {titles.get(doc_id, doc_id)}\n")
    if not result["edges"]:
        click.echo("  (no coupling/co-citation neighbors — paper may be sparsely keyed)")
        return
    for edge in result["edges"]:
        click.echo(
            f"  [{edge['weight']:>8.4f}] {titles.get(edge['dst'], edge['dst'])}  ({edge['dst']})"
        )


@map_group.command("seminal")
@click.argument("topic")
@click.option("--profile", default=None, help="Embed profile used to search for topic seeds.")
@click.option("--top", default=15, show_default=True, help="Anchor papers to return.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def map_seminal_cmd(topic: str, profile: str | None, top: int, as_json: bool) -> None:
    """Anchor papers for a topic: personalized PageRank + in-corpus cited-by counts.

    A discovery view over the citation graph — it never feeds retrieval ranking.
    """
    from graph_layout_rag.query.citation_rank import citation_prior, personalized_pagerank

    graph = _require_graph()
    seeds = _seed_doc_ids(topic, profile=profile)
    if not seeds:
        click.echo(f"No documents match {topic!r}.", err=True)
        sys.exit(1)
    seed_oas = {graph.doc_to_oa[d] for d in seeds if d in graph.doc_to_oa}
    if not seed_oas:
        click.echo(
            f"None of the {len(seeds)} matching doc(s) are keyed into the citation graph. "
            "Run: graph-layout-rag cite enrich",
            err=True,
        )
        sys.exit(1)

    ppr = personalized_pagerank(graph, seed_oas)
    scored = [
        (oa, score)
        for oa, score in ppr.items()
        if oa in graph.oa_to_doc  # corpus docs only: these are actionable anchors
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    titles = _titles()

    anchors = []
    for oa, score in scored[:top]:
        anchor_doc = graph.oa_to_doc[oa]
        in_corpus_citers = sum(
            1 for src in graph.in_adj.get(oa, set()) if src != oa and src in graph.oa_to_doc
        )
        anchors.append(
            {
                "doc_id": anchor_doc,
                "title": titles.get(anchor_doc, anchor_doc),
                "ppr": round(score, 6),
                "citation_prior": round(citation_prior(graph, oa), 4),
                "in_corpus_cited_by_count": in_corpus_citers,
                "seed": anchor_doc in seeds,
            }
        )

    result = {
        "schema_version": 1,
        "topic": topic,
        "seeds": seeds,
        "anchors": anchors,
    }
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"Anchor papers for {topic!r}:\n")
    for a in anchors:
        marker = "*" if a["seed"] else " "
        click.echo(
            f" {marker} [{a['in_corpus_cited_by_count']:>3} in-corpus cites, "
            f"ppr {a['ppr']:.4f}] {a['title']}  ({a['doc_id']})"
        )
    click.echo("\n(* = matched the topic query directly)")
