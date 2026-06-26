"""Multi-system pooling for de-biased relevance judging.

Builds a *diverse* candidate pool per gold case by unioning the top-k of several
retrievers (lexical, dense, hybrid, query-expansion, learned-sparse,
late-interaction), deduped on canonical document id. Judging this union — rather
than a single lexical system's output — removes the pooling bias that favored
BM25 in the original gold set (Buckley & Voorhees, *Bias and the Limits of
Pooling*; Lin et al. 2022).

Each pooled doc records which systems surfaced it and at what rank, so the
diagnostics step can show that the dense/ColBERT-only documents — the ones the
lexical-only pool missed — are now in the judged set.

The curated gold relevant docs are also folded into the pool (tagged
``curated``) so the judge grades them too; comparing the judge's grade on those
known-relevant docs is a free check on judge quality.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_layout_rag.eval.gold_validation import EVAL_TRACKS, EvalTrack, cases_for_track
from graph_layout_rag.eval.strategies import strategy_registry
from graph_layout_rag.manifest import load_manifest, manifest_by_id
from graph_layout_rag.paths import DATA_DIR

log = logging.getLogger("graph_layout_rag.eval.pooling")

# Diverse-by-construction: lexical, dense, hybrid fusion, LLM query expansion,
# learned-sparse, late-interaction. The point is to surface docs no single
# retrieval family would, especially the neural-only docs the BM25 pool missed.
DEFAULT_POOL_SYSTEMS: tuple[str, ...] = (
    "bm25",
    "dense",
    "hybrid",
    "hyde",
    "multi_query",
    "splade",
    "colbert",
)

# Which experimental index kind each strategy name consumes.
_EXPERIMENTAL_KIND = {"splade": "splade", "dense_splade": "splade", "colbert": "colbert"}

# --------------------------------------------------------------------------- #
# System provenance classification (for leave-dense-out / seed attribution)
# --------------------------------------------------------------------------- #
# A synthetic gold case "survives" only if the pooled judge graded its seed doc
# relevant — but the seed only reaches the judge with corroboration when a
# retrieval system surfaced it. If a *dense* (or HyDE/neural) system is what
# surfaced the seed, the case enters the gold set partly *because* dense found it,
# and the downstream bake-off then "discovers" dense beats BM25 on exactly those
# cases — a circularity. To measure and remove that, every pool system is tagged
# as lexical (BM25-family) vs dense/neural so a leave-dense-out gold subset can be
# recomputed.
#
# Classification rationale (how each was identified, from the strategy classes in
# ``eval/strategies/__init__.py``):
#   bm25        — pure lexical BM25 term matching (``name="bm25"``)        → LEXICAL
#   dense       — vector kNN over embeddings                              → DENSE
#   colbert     — late-interaction dense (token-level embeddings)         → DENSE
#   hybrid      — RRF fusion of dense + bm25; carries the dense signal     → DENSE
#   hyde        — LLM hypothetical-doc expansion → dense retrieval         → DENSE/NEURAL
#   multi_query — LLM query expansion → retrieval (neural rewrite)         → DENSE/NEURAL
#   splade      — learned-sparse neural model (trained term weights);
#                 although "sparse", it is neural-trained, not pure lexical → DENSE/NEURAL
LEXICAL_SEED_SYSTEMS: frozenset[str] = frozenset({"bm25"})
DENSE_NEURAL_SEED_SYSTEMS: frozenset[str] = frozenset(
    {"dense", "colbert", "hybrid", "hyde", "multi_query", "splade", "dense_splade"}
)


def classify_system(name: str) -> str:
    """Return ``"lexical"`` or ``"dense"`` (dense/neural) for a pool system name.

    Unknown systems default to ``"dense"`` (conservative: an unclassified system
    is assumed to carry neural signal, so it is excluded from the lexical-only
    seed set rather than silently treated as pure-lexical).
    """
    return "lexical" if name in LEXICAL_SEED_SYSTEMS else "dense"

POOL_DIR = DATA_DIR / "eval" / "pool"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_lookup() -> dict[str, Any]:
    return manifest_by_id(load_manifest())


def build_pool(
    track: EvalTrack,
    *,
    systems: list[str] | None = None,
    depth: int = 50,
    embed_profile: str,
    experimental_indexes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run each system at ``depth`` per case, union+dedupe, enrich for judging.

    ``experimental_indexes`` maps an index *kind* ("splade"/"colbert") to its
    built index directory. Requires a Docker Qdrant server
    (``GRAPH_RAG_QDRANT_URL``) for the late-interaction arm to avoid OOM.
    """
    if track not in EVAL_TRACKS:
        raise ValueError(f"Unknown eval track {track!r}")
    systems = systems or list(DEFAULT_POOL_SYSTEMS)
    experimental_indexes = experimental_indexes or {}
    registry = strategy_registry()
    by_id = _manifest_lookup()
    cases = cases_for_track(track)

    # Resolve which systems can actually run (experimental ones need an index dir).
    active: list[str] = []
    for name in systems:
        if name not in registry:
            raise ValueError(f"Unknown pool system {name!r}")
        kind = _EXPERIMENTAL_KIND.get(name)
        if kind and kind not in experimental_indexes:
            log.warning("skipping %s: no %s index dir provided", name, kind)
            continue
        active.append(name)

    case_pools: dict[str, Any] = {}
    for case in cases:
        pooled: dict[str, dict[str, Any]] = {}

        def _record(row: dict[str, Any], system: str, rank: int) -> None:
            canonical = row.get("canonical_doc_id") or row.get("doc_id") or ""
            if not canonical:
                return
            entry = pooled.setdefault(
                canonical,
                {
                    "canonical_doc_id": canonical,
                    "doc_id": row.get("doc_id"),
                    "title": row.get("title"),
                    "excerpt": row.get("excerpt"),
                    "source_url": row.get("source_url"),
                    "systems": {},
                    "curated": False,
                },
            )
            # Keep the best (lowest) rank each system gave the doc.
            prev = entry["systems"].get(system)
            entry["systems"][system] = rank if prev is None else min(prev, rank)
            if not entry.get("excerpt") and row.get("excerpt"):
                entry["excerpt"] = row["excerpt"]

        for name in active:
            strategy = registry[name]
            kind = _EXPERIMENTAL_KIND.get(name)
            if kind:
                os.environ["GRAPH_RAG_EXPERIMENTAL_INDEX"] = experimental_indexes[kind]
            try:
                rows = strategy.run(case, embed_profile=embed_profile, top=depth)
            except Exception as exc:  # noqa: BLE001 — one system failing shouldn't void the pool
                log.warning("pool system %s failed on %s: %s", name, case.id, exc)
                continue
            for rank, row in enumerate(rows, start=1):
                _record(row, name, rank)

        # Fold curated gold labels into the pool so the judge grades them too.
        for doc_id in case.relevant_doc_ids:
            canonical = doc_id
            entry = pooled.get(canonical)
            if entry is None:
                item = by_id.get(doc_id)
                entry = pooled.setdefault(
                    canonical,
                    {
                        "canonical_doc_id": canonical,
                        "doc_id": doc_id,
                        "title": item.title if item else None,
                        "excerpt": None,
                        "source_url": item.url if item else None,
                        "systems": {},
                        "curated": True,
                    },
                )
            entry["curated"] = True

        # Enrich every pooled doc with manifest title/abstract for the judge.
        for entry in pooled.values():
            item = by_id.get(entry["canonical_doc_id"]) or by_id.get(entry.get("doc_id") or "")
            if item:
                entry.setdefault("title", item.title)
                entry["title"] = entry.get("title") or item.title
                entry["abstract"] = item.abstract
                entry["year"] = item.year
                entry["doc_kind"] = item.documentKind
            else:
                entry["abstract"] = None

        case_pools[case.id] = {
            "query": case.query,
            "category": case.category,
            "pdf_only": case.pdf_only,
            "curated_relevant": sorted(case.relevant_doc_ids),
            "pooled": pooled,
        }
        log.info("pooled %s: %d candidates from %d systems", case.id, len(pooled), len(active))

    return {
        "version": 1,
        "generated_at": _now_iso(),
        "track": track,
        "depth": depth,
        "embed_profile": embed_profile,
        "systems": active,
        "case_count": len(cases),
        "cases": case_pools,
    }


def pool_path(track: EvalTrack) -> Path:
    return POOL_DIR / track / "pool.json"


def write_pool(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_pools(paths: list[str]) -> dict[str, Any]:
    """Merge multiple pool JSON files into a single pool payload.

    For each case ID that appears in any pool:
    - The candidate doc IDs are unioned across all pools.
    - The 'systems' provenance maps are unioned (if a doc appears in multiple
      pools, system names are merged and the best / lowest rank is kept).

    Returns a merged payload dict with the same schema as a single pool file.
    Raises ValueError if a path does not exist or the file is malformed.
    """
    if not paths:
        return {
            "version": 1,
            "generated_at": _now_iso(),
            "track": "merged",
            "depth": 0,
            "embed_profile": "merged",
            "systems": [],
            "case_count": 0,
            "cases": {},
        }

    loaded: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise ValueError(f"Pool file does not exist: {p!r}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to parse pool file {p!r}: {exc}") from exc
        if not isinstance(payload, dict) or "cases" not in payload:
            raise ValueError(f"Pool file {p!r} is malformed: missing 'cases' key")
        loaded.append(payload)

    # Collect union of all systems across all pools.
    all_systems: list[str] = []
    seen_sys: set[str] = set()
    for pl in loaded:
        for s in pl.get("systems", []):
            if s not in seen_sys:
                all_systems.append(s)
                seen_sys.add(s)

    merged_cases: dict[str, Any] = {}
    for pl in loaded:
        for case_id, case_data in pl.get("cases", {}).items():
            if case_id not in merged_cases:
                merged_cases[case_id] = {
                    "query": case_data.get("query"),
                    "category": case_data.get("category"),
                    "pdf_only": case_data.get("pdf_only", False),
                    "curated_relevant": sorted(case_data.get("curated_relevant", [])),
                    "pooled": {},
                }
            dest = merged_cases[case_id]["pooled"]
            for doc_id, entry in case_data.get("pooled", {}).items():
                if doc_id not in dest:
                    dest[doc_id] = {
                        k: v for k, v in entry.items() if k != "systems"
                    }
                    dest[doc_id]["systems"] = {}
                # Union the systems provenance: keep best (lowest) rank per system.
                for sys_name, rank in entry.get("systems", {}).items():
                    prev = dest[doc_id]["systems"].get(sys_name)
                    dest[doc_id]["systems"][sys_name] = (
                        rank if prev is None else min(prev, rank)
                    )
                # Propagate curated flag if set in any pool.
                if entry.get("curated"):
                    dest[doc_id]["curated"] = True

    return {
        "version": 1,
        "generated_at": _now_iso(),
        "track": "merged",
        "depth": max((pl.get("depth", 0) for pl in loaded), default=0),
        "embed_profile": "merged",
        "systems": all_systems,
        "case_count": len(merged_cases),
        "cases": merged_cases,
    }


def _seed_systems(entry: Any) -> set[str]:
    """System names that surfaced a pooled doc (handles dict or list ``systems``)."""
    if not isinstance(entry, dict):
        return set()
    systems = entry.get("systems", {})
    if isinstance(systems, dict):
        return set(systems.keys())
    if isinstance(systems, (list, tuple, set)):
        return set(systems)
    return set()


def seed_attribution(
    pool: dict[str, Any],
    case_seed: dict[str, str],
    *,
    qrels: dict[str, Any] | None = None,
    relevance_threshold: int = 2,
) -> dict[str, dict[str, Any]]:
    """Per-case seed-system attribution.

    For each gold case, report which pool system(s) *surfaced its seed doc* and a
    ``provenance`` tag (``"lexical"`` if any lexical system surfaced the seed,
    else ``"dense"`` if only dense/neural systems did, else ``"none"`` when no
    retrieval system surfaced the seed — i.e. it only entered the pool as a
    folded-in curated doc).

    When ``qrels`` is supplied, ``seed_grade`` and ``seed_corroborated`` (judge
    graded the seed >= ``relevance_threshold``) are filled in; the leave-dense-out
    subset uses ``seed_corroborated`` AND a lexical surfacer.

    Args:
        pool: A loaded pool payload (``{"cases": {case_id: {"pooled": {...}}}}``).
        case_seed: Mapping ``case_id -> seed_doc_id`` (from gold_synth cases.json).
        qrels: Optional loaded qrels payload to fill in judge grades.
        relevance_threshold: Grade at/above which the seed counts as corroborated.

    Returns:
        ``{case_id: {seed_doc_id, systems (sorted), provenance, lexical, dense,
        seed_grade, seed_corroborated}}``.
    """
    qrels_cases: dict[str, Any] = (qrels or {}).get("cases", {})
    out: dict[str, dict[str, Any]] = {}
    pool_cases = pool.get("cases", {})
    for case_id, seed_doc_id in case_seed.items():
        pooled = pool_cases.get(case_id, {}).get("pooled", {})
        systems = _seed_systems(pooled.get(seed_doc_id))
        lexical = {s for s in systems if classify_system(s) == "lexical"}
        dense = systems - lexical
        if lexical:
            provenance = "lexical"
        elif dense:
            provenance = "dense"
        else:
            provenance = "none"

        seed_grade: int | None = None
        if qrels_cases:
            entry = qrels_cases.get(case_id, {}).get(seed_doc_id)
            if isinstance(entry, dict) and entry.get("grade") is not None:
                seed_grade = int(entry["grade"])
        seed_corroborated = (
            seed_grade is not None and seed_grade >= relevance_threshold
        )

        out[case_id] = {
            "seed_doc_id": seed_doc_id,
            "systems": sorted(systems),
            "lexical": sorted(lexical),
            "dense": sorted(dense),
            "provenance": provenance,
            "seed_grade": seed_grade,
            "seed_corroborated": seed_corroborated,
        }
    return out


def attribution_breakdown(attribution: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarise a :func:`seed_attribution` map: how many cases per provenance.

    ``lexical_seeded`` counts cases a lexical (BM25-family) system surfaced;
    ``dense_only_seeded`` counts cases ONLY a dense/neural system surfaced (the
    circularity-prone cases); ``unseeded`` counts cases no retrieval system
    surfaced (folded-in curated seed only). ``hyde_only``/``dense_family_only``
    further break down the dense-only bucket for transparency.
    """
    lexical_seeded = dense_only = unseeded = 0
    corroborated_lexical = 0
    for info in attribution.values():
        prov = info["provenance"]
        if prov == "lexical":
            lexical_seeded += 1
            if info.get("seed_corroborated"):
                corroborated_lexical += 1
        elif prov == "dense":
            dense_only += 1
        else:
            unseeded += 1
    return {
        "n_cases": len(attribution),
        "lexical_seeded": lexical_seeded,
        "dense_only_seeded": dense_only,
        "unseeded": unseeded,
        "lexical_seeded_and_corroborated": corroborated_lexical,
    }


def leave_dense_out_case_ids(
    attribution: dict[str, dict[str, Any]],
    *,
    require_corroborated: bool = True,
) -> list[str]:
    """Gold case IDs that survive a LEAVE-DENSE-OUT pool.

    A case survives iff a *lexical* (BM25-family) system surfaced its seed — i.e.
    its inclusion does not depend on dense/HyDE/neural retrieval. When
    ``require_corroborated`` is True (default) and grades are available, the seed
    must also have been judged relevant. Cases seeded only by dense/neural systems
    (or unseeded) are excluded, breaking the circularity in the BM25-falls headline.
    """
    out: list[str] = []
    for case_id, info in attribution.items():
        if info["provenance"] != "lexical":
            continue
        if require_corroborated and info.get("seed_grade") is not None:
            if not info.get("seed_corroborated"):
                continue
        out.append(case_id)
    return sorted(out)


def pool_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Summary stats: total candidates, mean per case, and neural-only count.

    A 'neural-only' doc is one no lexical system (bm25/splade text-overlap) ranked
    but a dense/late-interaction system did — exactly the docs a BM25-only pool
    would have missed.
    """
    lexical = {"bm25", "splade"}
    total = 0
    neural_only = 0
    curated_outside_lexical = 0
    per_case: list[int] = []
    for case in payload["cases"].values():
        pooled = case["pooled"]
        per_case.append(len(pooled))
        total += len(pooled)
        for entry in pooled.values():
            surfacing = set(entry["systems"])
            if surfacing and not (surfacing & lexical):
                neural_only += 1
            if entry.get("curated") and surfacing and not (surfacing & lexical):
                curated_outside_lexical += 1
    return {
        "total_candidates": total,
        "mean_candidates_per_case": round(total / max(1, len(per_case)), 1),
        "neural_only_candidates": neural_only,
        "curated_docs_no_lexical_system": curated_outside_lexical,
    }
