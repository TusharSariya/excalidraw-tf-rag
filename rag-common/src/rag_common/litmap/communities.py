"""Leiden community detection over the fused kNN graph, with cluster labeling.

Partitioning uses `python-igraph` + `leidenalg` (native extensions — flagged as a lock-file
risk in the E1 plan) with a fixed `seed` for determinism. Leiden itself only guarantees a
connected partition; on a disconnected graph (common once bib-coupling and cosine kNN are
fused and thresholded) each connected component is partitioned independently so a
low-connectivity bridge between two topics never merges them into one cluster by accident.

Cluster labels default to extractive TF-IDF over member titles+tags (pure Python — no
sklearn dependency): term frequency within the cluster weighted by inverse document
frequency across the full corpus, so common corpus-wide words (e.g. "graph", "algorithm" in
a graph-layout corpus) don't dominate every label. `--llm-labels` swaps in
`rag_common.local_llm.generate_text`; any failure (backend down, empty response, exception)
falls back to the extractive label rather than propagating, so labeling never blocks
`map clusters`.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

SCHEMA_VERSION = 1

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "to",
    "with",
    "via",
    "using",
    "based",
    "study",
    "toward",
    "towards",
    "approach",
    "approaches",
    "method",
    "methods",
    "model",
    "models",
    "system",
    "systems",
    "analysis",
    "new",
    "novel",
    "paper",
    "survey",
    "review",
    "introduction",
    "part",
    "case",
    "into",
    "from",
    "over",
    "under",
    "between",
    "is",
    "are",
    "as",
    "by",
    "at",
    "we",
    "this",
    "that",
    "its",
    "their",
    "work",
}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")


def tokenize(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 2
    ]


def build_corpus_df(doc_texts: Iterable[str]) -> tuple[dict[str, int], int]:
    """Document frequency of each token across the whole corpus, for IDF weighting."""
    df: Counter[str] = Counter()
    n = 0
    for text in doc_texts:
        n += 1
        df.update(set(tokenize(text)))
    return dict(df), n


def extractive_label(
    member_texts: list[str],
    corpus_df: dict[str, int],
    corpus_size: int,
    *,
    top_terms: int = 3,
) -> str:
    """Top TF-IDF terms across a cluster's member texts (title + tags, joined)."""
    tf: Counter[str] = Counter()
    for text in member_texts:
        tf.update(tokenize(text))
    if not tf:
        return "unlabeled cluster"
    scored = []
    for term, count in tf.items():
        df = corpus_df.get(term, 1)
        idf = math.log((corpus_size + 1) / (df + 1)) + 1.0
        scored.append((term, count * idf))
    scored.sort(key=lambda t: t[1], reverse=True)
    top = [term for term, _ in scored[:top_terms]]
    return " / ".join(top) if top else "unlabeled cluster"


def llm_label(
    member_titles: list[str],
    *,
    fallback_texts: list[str],
    corpus_df: dict[str, int],
    corpus_size: int,
    model: str | None = None,
    max_titles: int = 15,
) -> tuple[str, str]:
    """LLM-generated short topic label, falling back to `extractive_label` on any failure
    (backend unreachable, empty/garbage response, exception of any kind)."""
    try:
        from rag_common.local_llm import generate_text

        listing = "\n".join(f"- {t}" for t in member_titles[:max_titles] if t)
        prompt = (
            "Give a short (2-6 word) topic label for a cluster of research papers with "
            "these titles. Respond with only the label, no punctuation or explanation.\n\n"
            f"{listing}"
        )
        text = generate_text(prompt, model=model, max_tokens=32)
        label = text.strip().splitlines()[0].strip().strip('"').strip(".")
        if not label:
            raise ValueError("empty LLM label")
        return label, "llm"
    except Exception:
        return extractive_label(fallback_texts, corpus_df, corpus_size), "extractive"


def leiden_communities(
    edges: list[tuple[str, str, float]],
    *,
    all_doc_ids: Iterable[str] | None = None,
    seed: int = 42,
    resolution: float = 1.0,
) -> dict[str, int]:
    """Partition doc ids into integer cluster ids via Leiden, per connected component.

    `all_doc_ids`, if given, ensures every doc gets a cluster id even if it has no fused
    neighbor edges (isolated docs each become their own singleton cluster) — otherwise only
    docs that appear in `edges` are assigned.
    """
    import igraph as ig
    import leidenalg

    nodes = sorted({n for u, v, _ in edges for n in (u, v)} | set(all_doc_ids or ()))
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph()
    g.add_vertices(len(nodes))
    if edges:
        g.add_edges([(idx[u], idx[v]) for u, v, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]

    membership = [-1] * len(nodes)
    next_cluster_id = 0
    for comp in g.components(mode="weak"):
        if len(comp) == 1:
            membership[comp[0]] = next_cluster_id
            next_cluster_id += 1
            continue
        sub = g.subgraph(comp)
        weights = sub.es["weight"] if "weight" in sub.edge_attributes() else None
        part = leidenalg.find_partition(
            sub,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            seed=seed,
            resolution_parameter=resolution,
        )
        local_to_global = max(part.membership) + 1
        for local_i, local_cluster in enumerate(part.membership):
            membership[comp[local_i]] = next_cluster_id + local_cluster
        next_cluster_id += local_to_global

    return {nodes[i]: membership[i] for i in range(len(nodes))}


def _doc_text(meta: dict) -> str:
    title = meta.get("title") or ""
    tags = meta.get("tags") or []
    return f"{title} {' '.join(tags)}"


@dataclass(frozen=True)
class Cluster:
    id: int
    doc_ids: list[str]
    size: int
    label: str
    label_method: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "size": self.size,
            "label": self.label,
            "label_method": self.label_method,
            "doc_ids": self.doc_ids,
        }


def build_clusters(
    edges: list[tuple[str, str, float]],
    doc_meta: dict[str, dict],
    *,
    all_doc_ids: Iterable[str] | None = None,
    seed: int = 42,
    resolution: float = 1.0,
    use_llm: bool = False,
    llm_model: str | None = None,
    top_terms: int = 3,
) -> dict:
    """Full clustering pipeline: partition + label. Returns the `map clusters` JSON schema.

    `doc_meta` maps doc_id -> `{"title": str, "tags": list[str]}` (missing entries default
    to empty text, which just yields a thin/unlabeled cluster rather than an error).
    """
    membership = leiden_communities(
        edges, all_doc_ids=all_doc_ids, seed=seed, resolution=resolution
    )
    groups: dict[int, list[str]] = defaultdict(list)
    for doc_id, cid in membership.items():
        groups[cid].append(doc_id)

    all_texts = [_doc_text(doc_meta.get(d, {})) for d in membership]
    corpus_df, corpus_size = build_corpus_df(all_texts)

    clusters: list[Cluster] = []
    for cid, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        members_sorted = sorted(members)
        texts = [_doc_text(doc_meta.get(d, {})) for d in members_sorted]
        if use_llm:
            titles = [doc_meta.get(d, {}).get("title") or d for d in members_sorted]
            label, method = llm_label(
                titles,
                fallback_texts=texts,
                corpus_df=corpus_df,
                corpus_size=corpus_size,
                model=llm_model,
            )
        else:
            label = extractive_label(texts, corpus_df, corpus_size, top_terms=top_terms)
            method = "extractive"
        clusters.append(
            Cluster(
                id=cid,
                doc_ids=members_sorted,
                size=len(members_sorted),
                label=label,
                label_method=method,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_count": len(clusters),
        "doc_count": len(membership),
        "clusters": [c.to_dict() for c in clusters],
    }
