# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regenerate the themed seed catalog under v3/seed/seeds/<corpus>/<theme>.toml.

Extracts every DOI / arxiv identity from a sibling RAG tool's harvest modules,
resolves a human title for each (Crossref, then OpenAlex), classifies it into a
research theme, and writes one TOML file per theme. AST-only parsing (no imports of
the target packages), so it has zero import side effects.

The generated TOMLs are the source of truth for `seed.py` — this script is the
convenience that (re)builds them when an upstream catalog grows. Files are
hand-editable; re-running overwrites the themes it produces.

    uv run v3/seed/build_seeds.py                 # rebuild all corpora
    uv run v3/seed/build_seeds.py --no-resolve    # skip network title lookups (fast)

Entry forms recognised across both tools: dict literals, bare DOI strings in a
list, `ManifestItem(...)` constructor calls, and the `_arxiv_pdf(arxiv_id, title,
year, tags, ...)` factory used by rag-literature's PDF seeds.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # excalidraw-tf-rag/
CACHE = HERE / ".title_cache.json"  # doi -> resolved title, so re-runs skip the network
MAILTO = "tushar.sariya77@gmail.com"
UA = f"harvest-seed-build/0.1 (mailto:{MAILTO})"

ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})")
ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", re.IGNORECASE)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


# ------------------------------------------------------------------ extraction


@dataclass
class Entry:
    kind: str  # "arxiv_id" | "doi"
    value: str
    title: str | None = None
    tags: list[str] = field(default_factory=list)


def _str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _str_list(node) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    return [s for e in node.elts if (s := _str(e)) is not None]


def _arxiv_id(fields: dict) -> str | None:
    if fields.get("arxiv_id"):
        return fields["arxiv_id"]
    for src in (fields.get("url", ""), fields.get("doi", "")):
        m = ARXIV_URL_RE.search(src) or ARXIV_DOI_RE.search(src)
        if m:
            return m.group(1)
    return None


def _entry_from_fields(fields: dict, tags: list[str]) -> Entry | None:
    title = fields.get("title") or fields.get("title_hint")
    ax = _arxiv_id(fields)
    if ax:
        return Entry("arxiv_id", ax, title, tags)
    doi = fields.get("doi")
    if doi and not ARXIV_DOI_RE.search(doi):
        return Entry("doi", doi, title, tags)
    return None


def extract(path: Path) -> list[Entry]:
    """Every seed identity in one harvest module, from all four entry forms."""
    out: list[Entry] = []
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        # dict literals anywhere: {"doi"/"arxiv_id"/"url"/"title"/"title_hint"/"tags": ...}
        if isinstance(node, ast.Dict):
            fields = {
                k: v
                for key, val in zip(node.keys, node.values)
                if (k := _str(key)) and (v := _str(val)) is not None
            }
            tags = next(
                (_str_list(val) for key, val in zip(node.keys, node.values) if _str(key) == "tags"),
                [],
            )
            if e := _entry_from_fields(fields, tags):
                out.append(e)
        # ManifestItem(...) and _arxiv_pdf(...) calls anywhere
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "ManifestItem":
                fields = {
                    kw.arg: v
                    for kw in node.keywords
                    if kw.arg and (v := _str(kw.value)) is not None
                }
                tags = next((_str_list(kw.value) for kw in node.keywords if kw.arg == "tags"), [])
                if e := _entry_from_fields(fields, tags):
                    out.append(e)
            elif name == "_arxiv_pdf" and len(node.args) >= 2:
                # positional: (arxiv_id, title, year, tags, ...)
                ax, title = _str(node.args[0]), _str(node.args[1])
                tags = _str_list(node.args[3]) if len(node.args) >= 4 else []
                if ax:
                    out.append(Entry("arxiv_id", ax, title, tags))
        # bare DOI strings that are direct elements of a list (e.g. ELK_REFERENCE_DOIS)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            getattr(node, "value", None), ast.List
        ):
            for el in node.value.elts:
                s = _str(el)
                if s and DOI_RE.match(s):
                    m = ARXIV_DOI_RE.search(s)
                    out.append(
                        Entry("arxiv_id", m.group(1), None, []) if m else Entry("doi", s, None, [])
                    )
    return out


# ------------------------------------------------------------------ titles


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


def _clean(title: str) -> str:
    for _ in range(3):
        title = html.unescape(title)
    return title.strip()


def resolve_title(doi: str) -> str | None:
    j = _get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}")
    if j and (titles := (j.get("message") or {}).get("title")):
        return _clean(titles[0])
    j = _get_json(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?mailto={MAILTO}")
    if j and (t := j.get("title") or j.get("display_name")):
        return _clean(t)
    return None


# ------------------------------------------------------------------ themes


def classify(entry: Entry, themes: list[tuple[str, list[str]]]) -> str:
    """First theme whose keyword matches the entry's tags or (resolved) title."""
    hay = " ".join(t.lower() for t in entry.tags)
    if entry.title:
        hay += " " + entry.title.lower()
    for name, keywords in themes:
        if any(kw in hay for kw in keywords):
            return name
    return "uncategorized"


# ------------------------------------------------------------------ corpus specs


@dataclass
class CorpusSpec:
    corpus: str
    harvest_dir: Path
    modules: list[str]
    themes: list[tuple[str, list[str]]]


GLR_THEMES = [
    (
        "neural-graph-drawing",
        [
            "gnn",
            "graph neural",
            "deepgd",
            "graph attention",
            "graphsage",
            "graph embedding",
            "graph-embeddings",
            "representation learning",
            "deepwalk",
            "node2vec",
            "network embedding",
            "social representations",
            "feature learning for networks",
        ],
    ),
    (
        "analytics",
        [
            "pagerank",
            "hits",
            "community",
            "louvain",
            "leiden",
            "modularity",
            "girvan",
            "centrality",
            "dynamic graph",
            "temporal",
            "graph-analytics",
            "spectral clustering",
            "normalized cut",
        ],
    ),
    (
        "routing-edges",
        [
            "routing",
            "orthogonal",
            "edge bundling",
            "edge-bundling",
            "connector",
            "libavoid",
            "hyperedge",
            "manhattan",
            "metro map",
        ],
    ),
    (
        "compaction-packing",
        [
            "compaction",
            "packing",
            "strip-packing",
            "floorplan",
            "sequence-pair",
            "overlap",
            "rectangle packing",
            "vlsi",
            "prism",
        ],
    ),
    (
        "constraints-compound",
        ["constraint", "vpsc", "compound", "cola", "ipsep", "separation", "containment", "grouped"],
    ),
    (
        "force-directed-stress",
        [
            "force-directed",
            "force directed",
            "spring",
            "stress",
            "kamada",
            "majorization",
            "multilevel",
            "sfdp",
            "mds",
            "fruchterman",
            "neato",
            "energy",
            "fisheye",
            "spectral",
        ],
    ),
    (
        "sugiyama-pipeline",
        [
            "sugiyama",
            "layered",
            "layering",
            "hierarchical",
            "crossing",
            "coordinate",
            "coffman",
            "dummy",
            "port",
            "rank",
            "dot ",
            "layer",
            "feedback arc",
            "dynadag",
            "incremental layout",
        ],
    ),
    (
        "trees-planar-geometry",
        [
            "tree",
            "planar",
            "planarity",
            "pq-tree",
            "visibility",
            "reingold",
            "tidy",
            "radial",
            "circular",
            "grid",
            "computational geometry",
            "arc diagram",
        ],
    ),
    (
        "foundational-books",
        ["book", "handbook", "survey", "overview", "aesthetic", "chapter", "proceedings"],
    ),
]

RAG_THEMES = [
    (
        "graphrag",
        [
            "graphrag",
            "graph-rag",
            "graph rag",
            "hipporag",
            "lightrag",
            "noderag",
            "catrag",
            "knowledge graph",
            "dual-level",
            "graph-based",
        ],
    ),
    (
        "agentic",
        [
            "agentic",
            "react",
            "toolformer",
            "webgpt",
            "self-ask",
            "auto-rag",
            "search-r1",
            "r1-searcher",
            "deepresearch",
            "webthinker",
            "research",
            "planning",
            "iterative-retrieval",
            "tool",
        ],
    ),
    (
        "self-correcting",
        [
            "self-rag",
            "self-correcting",
            "corrective",
            "crag",
            "reflection",
            "evaluator",
            "critique",
            "seakr",
            "self-aware",
        ],
    ),
    (
        "late-interaction-sparse",
        [
            "late-interaction",
            "late interaction",
            "colbert",
            "plaid",
            "splade",
            "sparse",
            "multi-vector",
            "jacolbert",
            "pylate",
        ],
    ),
    (
        "reranking",
        [
            "rerank",
            "rankgpt",
            "monot5",
            "rankrag",
            "rank1",
            "reasonrank",
            "cross-encoder",
            "listwise",
        ],
    ),
    (
        "chunking-indexing",
        [
            "chunk",
            "raptor",
            "late chunking",
            "contextual retrieval",
            "hierarchical-index",
            "indexing",
            "segmentation",
        ],
    ),
    (
        "dense-retrieval-embeddings",
        [
            "dense-retrieval",
            "dense retrieval",
            "dpr",
            "embedding",
            "sbert",
            "mteb",
            "matryoshka",
            "nomic",
            "jina",
            "contriever",
            "sentence",
            "bge",
        ],
    ),
    (
        "query-transformation",
        [
            "query-expansion",
            "query expansion",
            "query-rewriting",
            "rewrite",
            "hyde",
            "genread",
            "expansion",
        ],
    ),
    (
        "long-context",
        ["long-context", "long context", "lost in the middle", "longrag", "context window"],
    ),
    (
        "evaluation",
        [
            "evaluation",
            "benchmark",
            "beir",
            "ragas",
            "rgb",
            "frames",
            "bright",
            "freshstack",
            "litsearch",
            "umbrela",
            "llm-as-judge",
            "judge",
            "metrics",
            "robustness",
            "multi-hop",
        ],
    ),
    (
        "training-memory-compression",
        [
            "training",
            "atlas",
            "ra-dit",
            "fine-tun",
            "compression",
            "recomp",
            "filco",
            "chain-of-note",
            "memory",
            "memorag",
            "knn-lm",
            "retro",
            "distill",
        ],
    ),
    (
        "foundations",
        ["foundations", "realm", "fusion-in-decoder", "retrieval-augmented generation"],
    ),
]

SPECS = [
    CorpusSpec(
        "graph-layout-rag",
        REPO / "graph-layout-rag/src/graph_layout_rag/harvest",
        [
            "curated.py",
            "topic_seeds.py",
            "maintainer_seeds.py",
            "graphviz_theory.py",
            "elk_references.py",
            "handbook.py",
            "books.py",
        ],
        GLR_THEMES,
    ),
    CorpusSpec(
        "rag-literature-rag",
        REPO / "rag-literature-rag/src/rag_literature_rag/harvest",
        # only the RAG catalogs — the vestigial graphviz/elk/handbook modules are
        # graph-drawing leftovers from the fork and belong to the other corpus.
        ["topic_seeds.py", "curated.py"],
        RAG_THEMES,
    ),
]


# ------------------------------------------------------------------ emit


def toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def emit_corpus(spec: CorpusSpec, resolve: bool) -> None:
    # 1. extract + dedup (key on kind + normalised value; merge in any title/tags)
    seeds: dict[str, Entry] = {}
    for module in spec.modules:
        path = spec.harvest_dir / module
        if not path.exists():
            print(f"  ! missing module {path}")
            continue
        for e in extract(path):
            key = f"{e.kind}:{e.value.lower()}" if e.kind == "doi" else f"{e.kind}:{e.value}"
            if key not in seeds:
                seeds[key] = e
            else:
                cur = seeds[key]
                if e.title and not cur.title:
                    cur.title = e.title
                cur.tags = list({*cur.tags, *e.tags})

    entries = list(seeds.values())

    # 2. resolve missing titles (doi seeds only; arxiv titles come from the source).
    # A doi->title cache makes re-runs (e.g. theme tweaks) network-free.
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    misses = 0
    if resolve:
        for e in entries:
            if e.title or e.kind != "doi":
                continue
            if e.value in cache:
                e.title = cache[e.value]
                continue
            e.title = resolve_title(e.value)
            if e.title:
                cache[e.value] = e.title
            else:
                misses += 1
            time.sleep(0.4)
        CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))

    # 3. classify into themes
    by_theme: dict[str, list[Entry]] = {}
    for e in entries:
        by_theme.setdefault(classify(e, spec.themes), []).append(e)

    # 4. write one file per theme
    out_dir = HERE / "seeds" / spec.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme, group in sorted(by_theme.items()):
        group.sort(key=lambda e: (e.kind != "arxiv_id", (e.title or e.value).lower()))
        lines = [
            f"# {spec.corpus} — {theme}",
            "#",
            "# Generated by v3/seed/build_seeds.py from the graph-layout-rag /",
            "# rag-literature-rag harvest catalogs. Hand-editable; re-running the",
            "# generator overwrites this file. Only arxiv_id seeds hit the arxiv source;",
            "# doi-only seeds resolve via s2 (and openalex once wired).",
            "",
            f'corpus = "{spec.corpus}"',
            "",
        ]
        for e in group:
            lines.append("[[seed]]")
            if e.title:
                lines.append(f'title = "{toml_escape(e.title)}"')
            else:
                lines.append("# unresolved — no title from Crossref/OpenAlex")
            lines.append(f'{e.kind} = "{e.value}"')
            lines.append("")
        (out_dir / f"{theme}.toml").write_text("\n".join(lines))

    total = len(entries)
    ax = sum(1 for e in entries if e.kind == "arxiv_id")
    print(
        f"\n{spec.corpus}: {total} seeds ({ax} arxiv_id, {total - ax} doi), "
        f"{len(by_theme)} themes, {misses} title(s) unresolved"
    )
    for theme in sorted(by_theme):
        print(f"  {len(by_theme[theme]):3}  {theme}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-resolve", action="store_true", help="skip network title lookups")
    args = ap.parse_args()
    for spec in SPECS:
        emit_corpus(spec, resolve=not args.no_resolve)


if __name__ == "__main__":
    main()
