"""Relevance filter for graph drawing / layout literature.

Uses a small scoring model rather than a flat keyword-OR test. The old
``LAYOUT_KEYWORDS`` set passed any text containing a single generic word
(``graph``, ``network``, ``tree``, ``layout``, ``architecture``), which let in
the OpenAlex long tail (root architecture of boreal trees, neural networks,
wind-turbine dynamics, etc.). The scoring model requires either a
drawing/algorithm-specific *strong* term, or several generic *weak* terms that
co-occur with a visualization-context word.
"""

from __future__ import annotations

import re

# High-signal, hand-curated harvest sources. Items from these are always kept by
# the relevance gates (verify downgrade, prune) regardless of score — they are
# foundational seeds whose titles/abstracts may be terse.
CURATED_SOURCES = frozenset(
    {
        "graphviz.org",
        "handbook",
        "book",
        "topic-seed",
        "curated",
        "elk-bibliography",
        "research-thread",
        "kiel",
        "saarland",
        "monash",
        "ubc",
        "nsf-par",
        "springer",
        "wiley",
        "mermaid",
        "elk",
        "ogdf",
        "dagre",
        "yifanhu",
    }
)

# Strong, drawing/algorithm-specific terms. Any single one proves relevance.
STRONG_LAYOUT_TERMS = frozenset(
    {
        "graph drawing",
        "graph layout",
        "graph visualization",
        "graph visualisation",
        "node-link",
        "node link diagram",
        "sugiyama",
        "layered graph",
        "layered drawing",
        "layered layout",
        "layer assignment",
        "layer-assignment",
        "graph layering",
        "crossing minimization",
        "crossing minimisation",
        "crossing reduction",
        "crossing number",
        "force-directed",
        "force directed",
        "spring embedder",
        "spring-embedder",
        "stress majorization",
        "stress majorisation",
        "stress model",
        "kamada-kawai",
        "kamada kawai",
        "fruchterman",
        "orthogonal drawing",
        "orthogonal layout",
        "orthogonal graph",
        "edge routing",
        "connector routing",
        "edge bundling",
        "bend minimization",
        "bend minimisation",
        "coordinate assignment",
        "brandes köpf",
        "brandes-köpf",
        "brandes and köpf",
        "planar embedding",
        "planar drawing",
        "upward drawing",
        "upward planar",
        "level planar",
        "treemap",
        "tree drawing",
        "tidy tree",
        "tidier drawing",
        "metro map",
        "metro-map",
        "graphviz",
        "neato",
        "sfdp",
        "twopi",
        "dynagraph",
        "mermaid",
        "dagre",
        "cytoscape",
        "gephi",
        "eclipse layout kernel",
        "kieler",
        "elkjs",
        "network simplex",
        "symbolic layout",
        "layout compaction",
        "one-dimensional compaction",
        "two-dimensional compaction",
        "floorplan",
        "floorplanning",
        "sequence-pair",
        "sequence pair",
        "b*-tree",
        "slicing tree",
        "overlap removal",
        "node overlap",
        "mental map",
        "separation constraint",
        "vpsc",
        "ipsep-cola",
        "dig-cola",
        "topology-shape-metrics",
        "topology shape metrics",
        "coffman-graham",
        "coffman graham",
        "sankey",
        "circular layout",
        "radial layout",
        "hierarchical layout",
        "hierarchical drawing",
        "compound graph",
        "compound digraph",
        "clustered graph",
        "dag layout",
        "drawing directed graphs",
        "drawing of graphs",
        "drawing graphs",
        # --- graph analytics / network science (P1a expansion) ---
        # Named methods and their defining phrases are unambiguous subject markers:
        # a single hit proves the paper is in-domain. Broad ML terms
        # ("graph neural network", "representation learning") stay in the *weak*
        # set below so a generic deep-learning paper does not pass on them alone.
        "pagerank",
        "page rank",
        "personalized pagerank",
        "personalised pagerank",
        "hits algorithm",
        "hubs and authorities",
        "authoritative sources",
        "community detection",
        "community structure",
        "louvain",
        "leiden",
        "modularity optimization",
        "modularity optimisation",
        "modularity maximization",
        "modularity maximisation",
        "girvan-newman",
        "girvan newman",
        "graph partitioning",
        "network partitioning",
        "spectral clustering",
        "spectral partitioning",
        "graph laplacian",
        "normalized cut",
        "normalized cuts",
        "normalised cut",
        "normalised cuts",
        "fiedler vector",
        "laplacian eigenmap",
        "node2vec",
        "deepwalk",
        "node embedding",
        "graph embedding",
        "network embedding",
        # "edge bundling" already appears in the routing-area strong terms above.
        "edge bundles",
        "hierarchical edge bundling",
        "hierarchical edge bundles",
        "force-directed edge bundling",
        "orthogonal connector routing",
        "orthogonal edge routing",
        "hyperedge routing",
        "dynamic graph drawing",
        "dynamic graph visualization",
        "dynamic graph visualisation",
        "dynamic network visualization",
        "dynamic network visualisation",
        "temporal graph drawing",
    }
)

# Generic terms. Only count toward relevance when a drawing-context term is also
# present (see DRAWING_CONTEXT_TERMS), so e.g. "neural network" alone scores 0.
WEAK_LAYOUT_TERMS = frozenset(
    {
        "graph",
        "digraph",
        "network",
        "tree",
        "node",
        "vertex",
        "vertices",
        "edge",
        "diagram",
        "directed",
        "undirected",
        "cluster",
        "clustering",
        "hierarchical",
        "hierarchy",
        "topology",
        "layout",
        "drawing",
        "drawn",
        "layer",
        "layered",
        "planar",
        "embedding",
        "coordinates",
        "rank",
        "routing",
        "compaction",
        "packing",
        "overlap",
        "compound",
        "nested",
        "hypergraph",
        "dag",
        # --- graph-analytics / ML-on-graphs weak terms (P1a expansion) ---
        # Broad enough to appear in off-domain ML/network-science papers, so they
        # only score when a drawing/visualization-context word co-occurs. This is
        # deliberately how "graph neural network" is admitted: strong when the
        # paper is about *layout* ("...for graph drawing"), silent otherwise.
        "graph neural network",
        "graph convolutional",
        "graph attention",
        "representation learning",
        "community",
        "modularity",
        "centrality",
        "spectral",
        "laplacian",
        "eigenvector",
        "partitioning",
        "random walk",
        "temporal",
        "dynamic",
        "streaming",
    }
)

# A weak hit only counts when one of these visualization-context words appears.
DRAWING_CONTEXT_TERMS = frozenset(
    {
        "draw",
        "drawing",
        "drawn",
        "layout",
        "visual",
        "visualization",
        "visualisation",
        "diagram",
        "node-link",
        "aesthetic",
        "readability",
        "placement",
        "render",
        "graph drawing",
    }
)

# DOI prefixes that are definitionally never graph-layout venues.
# Checked before any API call so we never waste a quota request on a PLOS paper.
#
# P1a reconciliation (graph-analytics/network-science expansion): the corpus now
# wants community-detection, PageRank, and spectral classics, many of which live
# in *multidisciplinary megajournals* — Girvan-Newman (2002) is in PNAS, Newman's
# modularity work is in Physical Review E (10.1103, never blocked), Leiden (2019)
# is in Nature Scientific Reports, and PLOS ONE hosts network science. Blanket
# blocking those venues would reject the very seeds we are adding. So the venue
# block is narrowed to *definitionally biomedical* sub-journals, and the
# multidisciplinary megajournals (PNAS, Nature main/Comms, Scientific Reports,
# PLOS ONE) are released — the content-based OFF_TOPIC_KEYWORDS filter (genome,
# protein, clinical, …) plus the relevance score still guard against the
# biomedical long tail those venues also publish.
OFF_TOPIC_DOI_PREFIXES: tuple[str, ...] = (
    # PLOS: block the purely biomedical sub-journals; leave 10.1371/journal.pone
    # (PLOS ONE, multidisciplinary) to the content filter.
    "10.1371/journal.pbio",  # PLOS Biology
    "10.1371/journal.pmed",  # PLOS Medicine
    "10.1371/journal.pgen",  # PLOS Genetics
    "10.1371/journal.ppat",  # PLOS Pathogens
    "10.1371/journal.pcbi",  # PLOS Computational Biology
    "10.1371/journal.pntd",  # PLOS Neglected Tropical Diseases
    "10.1186/",  # BMC family (Bioinformatics, Systems Biology, Genome Biology…)
    # Nature: block the biomedical sub-journals; leave 10.1038/s41598 (Scientific
    # Reports), 10.1038/s41467 (Nature Communications), and Nature main open.
    "10.1038/nmeth",  # Nature Methods
    "10.1038/ng",  # Nature Genetics
    "10.1038/nbt",  # Nature Biotechnology
    "10.1038/nm",  # Nature Medicine
    "10.1038/nn",  # Nature Neuroscience
    "10.1038/nrg",  # Nature Reviews Genetics
    "10.1523/",  # Journal of Neuroscience
    "10.1016/j.neuroimage",
    "10.1016/j.brainres",
    "10.3389/fgene",  # Frontiers in Genetics
    "10.3389/fneu",  # Frontiers in Neuroscience / Neurology
    "10.3389/fnhu",  # Frontiers in Human Neuroscience
    "10.3389/fpls",  # Frontiers in Plant Science
    "10.3389/fpub",  # Frontiers in Public Health
    "10.3389/fpsyg",  # Frontiers in Psychology
    "10.3389/fmicb",  # Frontiers in Microbiology
    "10.1093/bioinformatics",
    "10.1093/nar/",  # Nucleic Acids Research
    "10.1093/hmg/",  # Human Molecular Genetics
    # NOTE: 10.1073 (PNAS) and 10.1126 (Science) are intentionally NOT blocked —
    # both are multidisciplinary and host network-science classics; the keyword
    # filter handles their biomedical majority.
)


def is_known_offtopic_doi(doi: str) -> bool:
    """Return True if the DOI prefix identifies a definitionally off-topic journal."""
    doi_lower = doi.lower()
    return any(doi_lower.startswith(p.lower()) for p in OFF_TOPIC_DOI_PREFIXES)


OFF_TOPIC_KEYWORDS = frozenset(
    {
        # biomedical / life sciences
        "genome",
        "genomic",
        "microbiome",
        "microbiota",
        "rna",
        "dna",
        "protein",
        "cancer",
        "tumor",
        "tumour",
        "oncology",
        "clinical trial",
        "patient",
        "hospital",
        "covid",
        "sars-cov",
        "vaccine",
        "epidemiology",
        "pathogen",
        "bacteria",
        "fungal",
        "metagenom",
        "transcriptom",
        "proteom",
        "proteomics",
        "chromosome",
        "hi-c",
        "sequencing",
        "phylogen",
        "ecology",
        "agriculture",
        "crop",
        "livestock",
        "boreal",
        "neurodegener",
        "alzheimer",
        "parkinson",
        "psychiatric",
        "depression",
        "schizophrenia",
        "cardiovascular",
        "diabetes",
        "obesity",
        "pharmacolog",
        "pharmacokinetic",
        "drug discovery",
        "drug-likeness",
        "medicinal",
        "medicinal chemistry",
        "microarray",
        "metabol",
        "dendritic",
        "dendrite",
        "synaptic",
        "spiking",
        "spiking neuron",
        # physics / materials / energy
        "quantum",
        "holographic",
        "metamaterial",
        "turbine",
        "wind energy",
        "floating wind",
        "parametron",
        "adiabatic",
        "photovoltaic",
        "plasma",
        "astrophys",
        "cosmolog",
        "seismic",
        "geophysic",
        # other off-topic CS / misc
        "machine learning algorithms real-world",
        "statistical computing",
        "bibliometric mapping",
        "vosviewer",
        "autoregressive time series",
        "variance-based structural",
        "vehicular network",
        "controller area network",
    }
)

# Off-topic keywords match at a *left* word boundary (so morphological variants
# like "phylogen" -> "phylogenetics" / "metabol" -> "metabolism" still fire) but
# NOT mid-word. A plain ``"rna" in hay`` substring test matched "exte`rna`l",
# "inte`rna`l", "jou`rna`l", "alte`rna`ting", "tou`rna`ment" — core graph-drawing
# vocabulary ("external face", "internal vertices") — and hard-killed those
# papers at discovery. The boundary anchor removes that false negative.
_OFF_TOPIC_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(OFF_TOPIC_KEYWORDS, key=len, reverse=True))
    + r")"
)


def is_off_topic(title: str, abstract: str | None = None) -> bool:
    """True if the title/abstract hits an off-topic keyword at a word boundary."""
    return bool(_OFF_TOPIC_RE.search(_haystack(title, abstract)))


def _haystack(title: str, abstract: str | None) -> str:
    return f"{title} {abstract or ''}".lower()


def _has_strong(hay: str) -> bool:
    return any(term in hay for term in STRONG_LAYOUT_TERMS)


def layout_relevance_score(title: str, abstract: str | None = None) -> int:
    """Score how graph-layout related a title/abstract is.

    +3 per strong term, +1 per weak term *only if* a drawing-context term is
    present, and a hard -100 if any off-topic keyword appears.
    """
    hay = _haystack(title, abstract)
    if _OFF_TOPIC_RE.search(hay):
        return -100
    score = 3 * sum(1 for term in STRONG_LAYOUT_TERMS if term in hay)
    if any(ctx in hay for ctx in DRAWING_CONTEXT_TERMS):
        score += sum(1 for term in WEAK_LAYOUT_TERMS if term in hay)
    return score


def is_pipeline_relevant(title: str, abstract: str | None = None) -> bool:
    """True when title/abstract matches a pipeline-layout category keyword."""
    # Lazy import to avoid a circular import (catalog -> classify -> relevance).
    from graph_layout_rag.catalog.taxonomy import categories_from_keywords

    return bool(categories_from_keywords(_haystack(title, abstract)))


def is_layout_relevant(
    title: str,
    abstract: str | None = None,
    *,
    strict: bool = False,
) -> bool:
    """Return True if title/abstract looks graph-layout related.

    - normal: score >= 3 (≈ at least one strong term, or several weak-in-context)
    - strict: at least one strong term present (and not off-topic).
    """
    hay = _haystack(title, abstract)
    score = layout_relevance_score(title, abstract)
    if strict:
        return score >= 3 and _has_strong(hay)
    return score >= 3
