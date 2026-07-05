"""Synthetic gold-case generation (DataMorgana / ARES style) for trustworthy eval.

The 49-case curated gold set is too small to *distinguish* retrieval methods: on it
every strategy clusters at ~0.90 nDCG@10 and any "win" is noise. This module grows
the judged eval to ~300 stratified cases so method deltas become statistically real.

Pipeline:
  1. ``sample_seed_docs`` — stratified, balanced sampling of manifest docs across
     category (tag-derived) × year-bucket, separately per track. Catalog seeds use
     title+abstract; pdf-deep-read seeds additionally pull a full-text chunk.
  2. ``generate_query`` — a *system-blind* local-LLM call writes an information
     need grounded in the seed. ``single`` mode = a direct need answerable by the
     seed; ``hard`` mode = indirect phrasing (applied, problem-first question).
  3. ``quality_filters`` — drop leakage (queries that lexically echo the seed →
     would inflate BM25), near-duplicate queries, and collisions with curated-49.
  4. ``build_synth_cases`` — assemble dicts and persist to
     ``data/eval/gold_synth/cases.json``.

The seed is only a *candidate* positive: final relevance comes from the existing
multi-system pooled judge (``eval pool`` → ``eval judge``), and a case whose seed
the judge later grades < 2 is dropped as a bad generation. So this module never
asserts relevance itself — it stays bias-honest by deferring to the pooled judge.

Bias guards live here by construction: system-blind prompts, an anti-lexical-leakage
filter, de-dup vs curated, and enforced stratum balance (reported in the manifest).

Multi-hop stratum (added 2026-07-04, P0.4): the corpus has since grown enough to
support a small, high-quality *relational* stratum. ``build_multihop_cases`` pairs
two thematically-linked docs — preferring a citation edge (via the persisted
citation graph), else a same-category partner — and prompts the LLM for one need
answerable ONLY by combining both. Both docs are marked relevant (seed grade 3,
partner grade 2), the case carries ``stratum="multi-hop"``, and the same
system-blind + anti-lexical-leakage + de-dup guards apply. These cases gate the
Phase-2 graph-arm eval, so they are kept deliberately small (quality over quantity)
and appended additively via ``append_synth_cases`` (never a full regeneration).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from graph_layout_rag.eval.gold_cases import GOLD_CASES
from graph_layout_rag.eval.gold_validation import EvalTrack
from graph_layout_rag.manifest import ManifestItem, load_manifest
from graph_layout_rag.paths import CHUNKS_TABLE, DATA_DIR, profile_index_paths

log = logging.getLogger("graph_layout_rag.eval.gold_synth")

# Bump when the generation prompt or filtering semantics change so a re-gen does
# not silently reuse stale cached queries.
SYNTH_VERSION = "synth-v1"

SYNTH_DIR = DATA_DIR / "eval" / "gold_synth"
CASES_PATH = SYNTH_DIR / "cases.json"
GEN_CACHE_PATH = SYNTH_DIR / "gen_cache.json"

GEN_WORKERS_ENV = "GRAPH_RAG_SYNTH_WORKERS"

MAX_ABSTRACT_CHARS = 1200
MAX_CHUNK_CHARS = 1200

# A query that re-uses too much of the seed's own wording leaks the answer to
# lexical retrievers. Reject if Jaccard token overlap with the seed exceeds this.
LEAKAGE_JACCARD_MAX = 0.45
# Two synthetic queries closer than this (token Jaccard) are near-duplicates.
DUP_JACCARD_MAX = 0.70

# Map the graph-layout corpus tag vocabulary onto curated category labels so
# synthetic strata line up with the hand-written gold set's categories.
_TAG_TO_CATEGORY: tuple[tuple[str, str], ...] = (
    # New-category needles first: more-specific tags (e.g. edge-routing) must
    # win over the generic needles below (e.g. routing).
    ("graph-analytics", "graph-analytics"),
    ("community-detection", "community-detection"),
    ("community detection", "community-detection"),
    ("spectral", "spectral"),
    ("graph-embeddings", "graph-embeddings"),
    ("graph-embedding", "graph-embeddings"),
    ("graph embedding", "graph-embeddings"),
    ("edge-routing", "edge-routing"),
    ("dynamic-graphs", "dynamic-graphs"),
    ("dynamic graph", "dynamic-graphs"),
    ("layer-assignment", "layer-assignment"),
    ("layer", "layer-assignment"),
    ("crossing", "crossing"),
    ("compound", "compound"),
    ("coordinate-assignment", "coordinate-assignment"),
    ("coordinate", "coordinate-assignment"),
    ("constraints", "constraints"),
    ("constraint", "constraints"),
    ("routing", "routing"),
    ("ports", "ports"),
    ("port", "ports"),
    ("packing", "packing"),
    ("overlap", "overlap"),
    ("compaction", "compaction"),
    ("compact", "compaction"),
    ("elk", "elk"),
)

_STOPWORDS = frozenset(
    "a an the of for to in on and or with via using how what why does do is are "
    "this that these those it its as at by from into over under more most can "
    "could would should when which who whom about across between within".split()
)


# --------------------------------------------------------------------------- #
# Stratification helpers
# --------------------------------------------------------------------------- #
def category_for(item: ManifestItem) -> str:
    """Derive a curated-vocabulary category from the doc's tags/title."""
    hay = " ".join(item.tags).lower() + " " + (item.title or "").lower()
    for needle, cat in _TAG_TO_CATEGORY:
        if needle in hay:
            return cat
    return "general"


def _year_bucket(year: int | None) -> str:
    if not year:
        return "unknown"
    if year <= 2010:
        return "<=2010"
    if year <= 2020:
        return "2011-2020"
    return ">=2021"


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Seed:
    doc_id: str
    title: str
    abstract: str
    chunk_text: str
    category: str
    year_bucket: str
    track: EvalTrack


# --------------------------------------------------------------------------- #
# Seed sampling
# --------------------------------------------------------------------------- #
def _fulltext_chunks(profile: str) -> dict[str, str]:
    """Longest stored chunk text per doc_id (for pdf-deep-read grounding)."""
    paths = profile_index_paths(profile)
    if not paths.lance_dir.exists():
        return {}
    import lancedb

    db = lancedb.connect(str(paths.lance_dir))
    names = db.list_tables()
    names = names if isinstance(names, list) else list(getattr(names, "tables", names))
    if CHUNKS_TABLE not in names:
        return {}
    tbl = db.open_table(CHUNKS_TABLE)
    rows = tbl.search().select(["doc_id", "text"]).limit(tbl.count_rows()).to_arrow().to_pylist()
    best: dict[str, str] = {}
    for r in rows:
        doc_id, text = r.get("doc_id"), r.get("text") or ""
        if doc_id and len(text) > len(best.get(doc_id, "")):
            best[doc_id] = text
    return best


def sample_seed_docs(
    profile: str,
    track: EvalTrack,
    n: int,
    *,
    rng: random.Random,
    exclude_ids: set[str],
) -> list[Seed]:
    """Stratified, balanced sampling across category × year-bucket for one track."""
    manifest = load_manifest()
    chunks = _fulltext_chunks(profile) if track == "pdf-deep-read" else {}

    candidates: list[Seed] = []
    for item in manifest.items:
        if item.id in exclude_ids:
            continue
        if not item.abstract:
            continue
        if track == "pdf-deep-read":
            # Needs real full text: status ok + a local PDF + a stored chunk.
            if not (item.status == "ok" and item.localPath and item.id in chunks):
                continue
        candidates.append(
            Seed(
                doc_id=item.id,
                title=item.title or "",
                abstract=(item.abstract or "")[:MAX_ABSTRACT_CHARS],
                chunk_text=chunks.get(item.id, "")[:MAX_CHUNK_CHARS],
                category=category_for(item),
                year_bucket=_year_bucket(item.year),
                track=track,
            )
        )

    # Group into strata and round-robin draw so no category/year dominates.
    strata: dict[tuple[str, str], list[Seed]] = defaultdict(list)
    for s in candidates:
        strata[(s.category, s.year_bucket)].append(s)
    for bucket in strata.values():
        rng.shuffle(bucket)

    keys = list(strata.keys())
    rng.shuffle(keys)
    picked: list[Seed] = []
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k]:
                picked.append(strata[k].pop())
                if len(picked) >= n:
                    break
    log.info("track=%s sampled %d/%d seeds across %d strata", track, len(picked), n, len(keys))
    return picked


# --------------------------------------------------------------------------- #
# Query generation (system-blind)
# --------------------------------------------------------------------------- #
def _gen_model() -> str:
    # Honour the active local_llm backend so the generator and judge share the
    # same transport (opencode/ollama/gateway).
    from rag_common.local_llm import llm_backend, ollama_model, opencode_model

    backend = llm_backend()
    if backend == "opencode":
        return opencode_model()
    return ollama_model()


def _gen_workers() -> int:
    raw = os.getenv(GEN_WORKERS_ENV, "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _load_cache() -> dict[str, str]:
    if not GEN_CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(GEN_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    GEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GEN_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    os.replace(tmp, GEN_CACHE_PATH)


def _cache_key(seed: Seed, mode: str) -> str:
    h = hashlib.sha256(seed.doc_id.encode()).hexdigest()[:16]
    return f"{SYNTH_VERSION}:{mode}:{h}"


def _build_prompt(seed: Seed, mode: str) -> str:
    body = seed.chunk_text or seed.abstract
    common = (
        "You write realistic search queries that a researcher would type into a "
        "retrieval-augmented generation (RAG) / neural-IR graph drawing literature "
        "search engine.\n"
        "Write the QUERY as an information NEED — describe what the searcher wants "
        "to find, do NOT copy the paper's title or distinctive phrasing (a good "
        "query is answerable by the paper but phrased independently).\n"
        "Output ONLY the query text, one line, 6-18 words, no quotes, no preamble.\n"
    )
    if mode == "hard":
        return (
            common + "Phrase it indirectly: an applied, problem-first question whose solution "
            "is this paper's method, using everyday wording rather than the paper's terms.\n"
            f"PAPER: {seed.title}\n{body}\n"
        )
    return common + f"PAPER: {seed.title}\n{body}\n"


def _generate_one(seed: Seed, mode: str, model: str) -> str:
    from rag_common.local_llm import generate_text

    prompt = _build_prompt(seed, mode)
    text = (generate_text(prompt, model=model, temperature=0.7, max_tokens=128) or "").strip()
    text = text.replace("\n", " ")
    return re.sub(r'^["\']|["\']$', "", text).strip()


def generate_queries(seeds: list[Seed], modes: list[str]) -> dict[int, str]:
    """Generate one query per (seed, mode); cache hits skip the LLM. Index-keyed."""
    cache = _load_cache()
    model = _gen_model()
    work = [(i, s, m) for i, (s, m) in enumerate(zip(seeds, modes))]
    misses = [(i, s, m) for i, s, m in work if _cache_key(s, m) not in cache]

    def _run(item: tuple[int, Seed, str]) -> tuple[int, str]:
        i, s, m = item
        try:
            return i, _generate_one(s, m, model)
        except Exception as exc:  # noqa: BLE001
            log.warning("synth gen failed for %s/%s (%s)", s.doc_id, m, exc)
            return i, ""

    if misses:
        done = 0
        with ThreadPoolExecutor(max_workers=_gen_workers()) as pool:
            for i, text in pool.map(_run, misses):
                if text:
                    cache[_cache_key(seeds[i], modes[i])] = text
                done += 1
                if done % 100 == 0:
                    _save_cache(cache)
                    log.info("synth gen %d/%d", done, len(misses))
        _save_cache(cache)
        log.info("synth gen: %d new queries", len(misses))

    return {i: cache.get(_cache_key(s, m), "") for i, s, m in work}


# --------------------------------------------------------------------------- #
# Quality filters
# --------------------------------------------------------------------------- #
@dataclass
class FilterStats:
    empty: int = 0
    leakage: int = 0
    duplicate: int = 0
    too_short: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "empty": self.empty,
            "leakage": self.leakage,
            "duplicate": self.duplicate,
            "too_short": self.too_short,
            "kept": self.kept,
        }


def quality_filters(
    seeds: list[Seed], modes: list[str], queries: dict[int, str]
) -> tuple[list[tuple[Seed, str, str]], FilterStats]:
    """Return surviving (seed, mode, query) triples + drop stats.

    Drops empty gens, lexical-leakage queries (echo the seed), near-duplicate
    queries, and too-short queries. Relevance is NOT decided here — that is the
    pooled judge's job downstream.
    """
    stats = FilterStats()
    curated_tokens = [_tokens(c.query) for c in GOLD_CASES]
    kept: list[tuple[Seed, str, str]] = []
    kept_tokens: list[set[str]] = []
    for i, seed in enumerate(seeds):
        q = queries.get(i, "").strip()
        mode = modes[i]
        if not q:
            stats.empty += 1
            continue
        if len(q.split()) < 4:
            stats.too_short += 1
            continue
        qt = _tokens(q)
        seed_tokens = _tokens(seed.title + " " + seed.abstract)
        if _jaccard(qt, seed_tokens) > LEAKAGE_JACCARD_MAX:
            stats.leakage += 1
            continue
        if any(_jaccard(qt, kt) > DUP_JACCARD_MAX for kt in kept_tokens):
            stats.duplicate += 1
            continue
        if any(_jaccard(qt, ct) > DUP_JACCARD_MAX for ct in curated_tokens):
            stats.duplicate += 1
            continue
        kept.append((seed, mode, q))
        kept_tokens.append(qt)
    stats.kept = len(kept)
    return kept, stats


# --------------------------------------------------------------------------- #
# Case assembly + persistence
# --------------------------------------------------------------------------- #
def _case_id(seed: Seed, mode: str, ordinal: int) -> str:
    return f"synth-{seed.track}-{mode}-{ordinal:04d}"


def build_synth_cases(
    profile: str,
    *,
    n_catalog: int,
    n_pdf: int,
    hard_frac: float = 0.30,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full generation pipeline → (case dicts, manifest). Does not judge."""
    rng = random.Random(seed)
    curated_ids = {d for c in GOLD_CASES for d in c.relevant_doc_ids}
    all_cases: list[dict[str, Any]] = []
    per_track_stats: dict[str, Any] = {}

    for track, n in (("catalog", n_catalog), ("pdf-deep-read", n_pdf)):
        if n <= 0:
            continue
        track_t: EvalTrack = track  # type: ignore[assignment]
        seeds = sample_seed_docs(profile, track_t, n, rng=rng, exclude_ids=set(curated_ids))
        n_hard = int(round(len(seeds) * hard_frac))
        modes = ["hard"] * n_hard + ["single"] * (len(seeds) - n_hard)
        rng.shuffle(modes)

        queries = generate_queries(seeds, modes)
        kept, fstats = quality_filters(seeds, modes, queries)

        for ordinal, (s, m, q) in enumerate(kept):
            all_cases.append(
                {
                    "id": _case_id(s, m, ordinal),
                    "query": q,
                    "relevant_ids": [s.doc_id],
                    "source": "synth",
                    "category": s.category,
                    "track": track,
                    "mode": m,
                    "seed_doc_id": s.doc_id,
                    "notes": f"synthetic;mode={m};seed={s.doc_id}",
                }
            )
        per_track_stats[track] = {
            "requested": n,
            "seeds": len(seeds),
            "hard": n_hard,
            "filters": fstats.to_dict(),
            "categories": dict(Counter(s.category for s in seeds)),
        }

    manifest = {
        "version": SYNTH_VERSION,
        "embed_profile": profile,
        "hard_frac": hard_frac,
        "seed": seed,
        "total_cases": len(all_cases),
        "by_track": per_track_stats,
        "by_category": dict(Counter(c["category"] for c in all_cases)),
        "by_mode": dict(Counter(c["mode"] for c in all_cases)),
    }
    return all_cases, manifest


def write_cases(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "cases": cases}
    tmp = CASES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, CASES_PATH)


def load_synth_cases() -> list[dict[str, Any]]:
    """Load persisted synthetic cases (empty list if none generated yet).

    Returns a list of dicts, each with keys:
        id, query, relevant_ids, source="synth"
    plus optional: category, track, mode, seed_doc_id, notes.

    Used by the opt-in merge in gold_cases() when GRAPH_RAG_INCLUDE_SYNTH=1.
    """
    if not CASES_PATH.is_file():
        return []
    try:
        payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    out: list[dict[str, Any]] = []
    for c in payload.get("cases", []):
        out.append(
            {
                "id": c["id"],
                "query": c["query"],
                "relevant_ids": list(c.get("relevant_ids", [])),
                "source": c.get("source", "synth"),
                "category": c.get("category"),
                "track": c.get("track"),
                "mode": c.get("mode"),
                "stratum": c.get("stratum", "single-hop"),
                "seed_doc_id": c.get("seed_doc_id"),
                "notes": c.get("notes", ""),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Multi-hop / relational stratum
# --------------------------------------------------------------------------- #
MULTIHOP_STRATUM = "multi-hop"
# Seed doc is the primary answer (grade 3), the paired partner is supporting (2).
MULTIHOP_SEED_GRADE = 3
MULTIHOP_PARTNER_GRADE = 2


def _citation_neighbors(graph: Any, doc_id: str) -> set[str]:
    """doc_ids that share a citation edge with ``doc_id`` (either direction).

    Returns an empty set when the graph is unavailable or the doc is not mapped
    onto the citation graph (only ~a third of corpus ids bridge to OpenAlex).
    """
    if graph is None:
        return set()
    oa = graph.doc_to_oa.get(doc_id)
    if not oa:
        return set()
    out: set[str] = set()
    for nb in graph.undirected.get(oa, ()):
        d = graph.oa_to_doc.get(nb)
        if d and d != doc_id:
            out.add(d)
    return out


def _load_citation_graph() -> Any:
    """Best-effort citation graph load; None if the store is missing/unreadable."""
    try:
        from graph_layout_rag.query.citation_rank import load_graph_cached

        return load_graph_cached()
    except Exception as exc:  # noqa: BLE001 — degrade to co-category pairing
        log.warning("multihop: citation graph unavailable (%s); using co-category pairs", exc)
        return None


def _multihop_prompt(a: Seed, b: Seed) -> str:
    body_a = (a.chunk_text or a.abstract)[:MAX_ABSTRACT_CHARS]
    body_b = (b.chunk_text or b.abstract)[:MAX_ABSTRACT_CHARS]
    return (
        "You write realistic search queries that a researcher would type into a "
        "graph drawing / layout literature search engine.\n"
        "Write ONE question whose full answer requires BOTH papers below — a "
        "relational or lineage need (how one technique relates to, builds on, "
        "refines, generalizes, or contrasts with the other). Do NOT name either "
        "paper's title or copy its distinctive phrasing; phrase it as an "
        "independent information need answerable only by combining the two.\n"
        "Output ONLY the query text, one line, 8-24 words, no quotes, no preamble.\n"
        f"PAPER A: {a.title}\n{body_a}\n\n"
        f"PAPER B: {b.title}\n{body_b}\n"
    )


def _pick_partner(
    seed: Seed,
    by_id: dict[str, Seed],
    by_cat: dict[str, list[Seed]],
    graph: Any,
    used: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[Seed, str] | None:
    """Pick a partner doc for ``seed``: citation neighbour first, else co-category.

    Returns ``(partner, pairing)`` where pairing is ``"citation"`` or
    ``"co-category"``; None when no eligible unused partner exists.
    """
    cited = [by_id[d] for d in _citation_neighbors(graph, seed.doc_id) if d in by_id]
    rng.shuffle(cited)
    cocat = [s for s in by_cat.get(seed.category, ()) if s.doc_id != seed.doc_id]
    rng.shuffle(cocat)
    for cand, pairing in [(c, "citation") for c in cited] + [(c, "co-category") for c in cocat]:
        key = tuple(sorted((seed.doc_id, cand.doc_id)))
        if key in used:
            continue
        return cand, pairing
    return None


def _multihop_cache_key(a: Seed, b: Seed) -> str:
    h = hashlib.sha256(f"{a.doc_id}|{b.doc_id}".encode()).hexdigest()[:16]
    return f"{SYNTH_VERSION}:multihop:{h}"


def _generate_multihop(pairs: list[tuple[Seed, Seed, str]], model: str) -> dict[int, str]:
    """Generate one relational query per pair; cache hits skip the LLM."""
    from rag_common.local_llm import generate_text

    cache = _load_cache()
    misses = [
        (i, a, b) for i, (a, b, _p) in enumerate(pairs) if _multihop_cache_key(a, b) not in cache
    ]

    def _run(item: tuple[int, Seed, Seed]) -> tuple[int, str]:
        i, a, b = item
        try:
            text = (
                generate_text(_multihop_prompt(a, b), model=model, temperature=0.7, max_tokens=128)
                or ""
            ).strip()
            text = text.replace("\n", " ")
            return i, re.sub(r'^["\']|["\']$', "", text).strip()
        except Exception as exc:  # noqa: BLE001 — degrade, retry next run
            log.warning("multihop gen failed for %s|%s (%s)", a.doc_id, b.doc_id, exc)
            return i, ""

    if misses:
        with ThreadPoolExecutor(max_workers=_gen_workers()) as pool:
            for i, text in pool.map(_run, misses):
                if text:  # only cache successes so 429s retry, not embed empties
                    a, b, _p = pairs[i]
                    cache[_multihop_cache_key(a, b)] = text
        _save_cache(cache)
        log.info("multihop gen: %d new queries", len(misses))

    return {i: cache.get(_multihop_cache_key(a, b), "") for i, (a, b, _p) in enumerate(pairs)}


def build_multihop_cases(
    profile: str,
    n: int,
    *,
    track: EvalTrack = "catalog",
    seed: int = 1234,
    oversample: int = 6,
    rng: random.Random | None = None,
    seeds: list[Seed] | None = None,
    graph: Any = None,
    exclude_queries: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate up to ``n`` multi-hop (relational) cases. Does not judge.

    ``seeds``/``graph`` may be injected (used by tests); otherwise seeds are
    sampled stratified from the manifest and the citation graph is loaded from
    disk. Both source docs are recorded relevant with grades; the surviving
    cases carry ``stratum="multi-hop"``. Returns ``(case dicts, stats)``.
    """
    rng = rng or random.Random(seed)
    if seeds is None:
        curated_ids = {d for c in GOLD_CASES for d in c.relevant_doc_ids}
        seeds = sample_seed_docs(
            profile, track, n * oversample, rng=rng, exclude_ids=set(curated_ids)
        )
    if graph is None:
        graph = _load_citation_graph()

    by_id = {s.doc_id: s for s in seeds}
    by_cat: dict[str, list[Seed]] = defaultdict(list)
    for s in seeds:
        by_cat[s.category].append(s)

    used: set[tuple[str, str]] = set()
    pairs: list[tuple[Seed, Seed, str]] = []
    pair_cap = max(n * 3, n + 8)  # oversample pairs to survive filtering
    for s in seeds:
        if len(pairs) >= pair_cap:
            break
        picked = _pick_partner(s, by_id, by_cat, graph, used, rng)
        if picked is None:
            continue
        partner, pairing = picked
        used.add(tuple(sorted((s.doc_id, partner.doc_id))))
        pairs.append((s, partner, pairing))

    model = _gen_model()
    queries = _generate_multihop(pairs, model)

    stats = FilterStats()
    curated_tokens = [_tokens(c.query) for c in GOLD_CASES]
    exclude_tokens = [_tokens(q) for q in (exclude_queries or [])]
    kept_tokens: list[set[str]] = []
    cases: list[dict[str, Any]] = []
    citation_linked = 0
    for i, (a, b, pairing) in enumerate(pairs):
        if len(cases) >= n:
            break
        q = (queries.get(i) or "").strip()
        if not q:
            stats.empty += 1
            continue
        if len(q.split()) < 6:
            stats.too_short += 1
            continue
        qt = _tokens(q)
        combined = _tokens(f"{a.title} {a.abstract} {b.title} {b.abstract}")
        if _jaccard(qt, combined) > LEAKAGE_JACCARD_MAX:
            stats.leakage += 1
            continue
        if any(
            _jaccard(qt, t) > DUP_JACCARD_MAX for t in kept_tokens + curated_tokens + exclude_tokens
        ):
            stats.duplicate += 1
            continue
        citation_linked += int(pairing == "citation")
        cases.append(
            {
                "id": f"synth-{track}-multihop-{len(cases):04d}",
                "query": q,
                "relevant_ids": [a.doc_id, b.doc_id],
                "source": "synth",
                "category": a.category,
                "track": track,
                "mode": "hard",
                "stratum": MULTIHOP_STRATUM,
                "seed_doc_id": a.doc_id,
                "partner_doc_id": b.doc_id,
                "pairing": pairing,
                "grades": {a.doc_id: MULTIHOP_SEED_GRADE, b.doc_id: MULTIHOP_PARTNER_GRADE},
                "notes": f"synthetic;stratum=multi-hop;pairing={pairing};seed={a.doc_id};partner={b.doc_id}",
            }
        )
        kept_tokens.append(qt)
    stats.kept = len(cases)
    manifest = {
        "stratum": MULTIHOP_STRATUM,
        "requested": n,
        "pairs": len(pairs),
        "citation_linked": citation_linked,
        "filters": stats.to_dict(),
        "categories": dict(Counter(c["category"] for c in cases)),
    }
    return cases, manifest


def append_synth_cases(new_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Additively append ``new_cases`` to cases.json — never regenerates existing.

    Existing cases are preserved verbatim; only cases with a fresh ``id`` are
    added. The manifest's ``total_cases``/``by_stratum`` counters are refreshed
    and an ``appended`` tally is recorded. Returns a small summary dict.
    """
    if CASES_PATH.is_file():
        payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    else:
        payload = {"manifest": {"version": SYNTH_VERSION}, "cases": []}
    existing: list[dict[str, Any]] = payload.get("cases", [])
    have = {c.get("id") for c in existing}
    added = [c for c in new_cases if c.get("id") not in have]
    payload["cases"] = existing + added
    manifest = payload.setdefault("manifest", {})
    manifest["total_cases"] = len(payload["cases"])
    manifest["by_stratum"] = dict(Counter(c.get("stratum", "single-hop") for c in payload["cases"]))
    manifest["appended"] = int(manifest.get("appended", 0)) + len(added)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CASES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, CASES_PATH)
    return {
        "added": len(added),
        "total": len(payload["cases"]),
        "skipped_existing": len(new_cases) - len(added),
    }
