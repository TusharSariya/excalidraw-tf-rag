"""Synthetic gold-case generation (DataMorgana / ARES style) for trustworthy eval.

The 49-case curated gold set is too small to *distinguish* retrieval methods: on it
every strategy clusters at ~0.90 nDCG@10 and any "win" is noise. This module grows
the judged eval to ~300 stratified cases so method deltas become statistically real.

Pipeline:
  1. ``sample_seed_docs`` — stratified, balanced sampling of manifest docs across
     category (tag-derived) × year-bucket, separately per track. Catalog seeds use
     title+abstract; pdf-deep-read seeds additionally pull a full-text chunk.
  2. ``generate_query`` — a *system-blind* Gemini-Flash call writes an information
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

Multi-hop is intentionally NOT used: the graph-layout corpus is too small for
reliable multi-hop; only single-hop NL paraphrases are generated.
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
from pathlib import Path
from typing import Any

from graph_layout_rag.eval.gold_cases import GOLD_CASES, EvalCase
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

GEN_MODEL_ENV = "GRAPH_RAG_SYNTH_LLM_MODEL"
DEFAULT_GEN_MODEL = "gemini-2.5-flash"
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
    rows = (
        tbl
        .search()
        .select(["doc_id", "text"])
        .limit(tbl.count_rows())
        .to_arrow()
        .to_pylist()
    )
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
    return os.getenv(GEN_MODEL_ENV, DEFAULT_GEN_MODEL).strip() or DEFAULT_GEN_MODEL


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
            common
            + "Phrase it indirectly: an applied, problem-first question whose solution "
            "is this paper's method, using everyday wording rather than the paper's terms.\n"
            f"PAPER: {seed.title}\n{body}\n"
        )
    return common + f"PAPER: {seed.title}\n{body}\n"


def _generate_one(seed: Seed, mode: str, model: str) -> str:
    import time

    from rag_common.gemini_embed import (
        _client,
        _is_fatal,
        _is_rate_limit,
        _parse_retry_after,
        llm_location,
    )

    client = _client(location=llm_location())
    prompt = _build_prompt(seed, mode)
    config: Any = None
    try:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.7)
    except Exception:  # noqa: BLE001
        config = None

    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            if config is not None:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
            else:
                resp = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip().replace("\n", " ")
            return re.sub(r'^["\']|["\']$', "", text).strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_fatal(exc) or not _is_rate_limit(exc):
                raise
            wait = _parse_retry_after(exc) or (2 ** attempt)
            time.sleep(min(wait, 30))
    raise last_exc if last_exc else RuntimeError("generation failed")


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
        seeds = sample_seed_docs(
            profile, track_t, n, rng=rng, exclude_ids=set(curated_ids)
        )
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
                "seed_doc_id": c.get("seed_doc_id"),
                "notes": c.get("notes", ""),
            }
        )
    return out
