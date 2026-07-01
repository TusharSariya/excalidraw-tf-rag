"""Recall-ceiling / corpus-coverage decomposition for the gold set.

A low recall@k has two very different causes with opposite fixes:

* the relevant doc is **not in the index** → a corpus-acquisition problem
  (harvest more full text; no ranker can retrieve what isn't there); or
* the relevant doc **is indexed** but the ranker doesn't surface it → a
  ranking problem (embeddings / fusion / reranking).

This command classifies every gold-relevant doc into a stratum and reports the
**recall ceiling** = fraction of relevant docs that are recoverable at all
(indexed). Comparing that ceiling with the actual recall@k from
``eval retrieval`` / ``eval benchmark`` tells you which lever to pull. It runs
no retrieval and needs no gateway/credentials — it only reads the index's
doc-id set and the manifest.
"""

from __future__ import annotations

import collections
import json
from typing import Any

import click

from graph_layout_rag.eval import corpus_health
from graph_layout_rag.eval.gold_validation import cases_for_track

# Strata a relevant doc can fall into. The first three are recoverable (present
# in the index); "not_indexed" is the hard corpus gap.
RECOVERABLE = ("full_text", "metadata_only", "indexed_unkeyed")


def _indexed_doc_ids(profile: str) -> set[str]:
    return {r["doc_id"] for r in corpus_health._load_chunks(profile)}


def _stratum(doc_id: str, indexed: set[str], by_id: dict[str, Any]) -> str:
    if doc_id not in indexed:
        return "not_indexed"
    item = by_id.get(doc_id)
    if item is None:
        return "indexed_unkeyed"
    if getattr(item, "status", None) == "ok" and getattr(item, "localPath", None):
        return "full_text"
    if getattr(item, "status", None) == "metadata_only":
        return "metadata_only"
    return getattr(item, "status", None) or "unknown"


def compute_coverage(profile: str, *, track: str = "catalog") -> dict[str, Any]:
    cases = cases_for_track(track)
    indexed = _indexed_doc_ids(profile)
    try:
        by_id = corpus_health._manifest_by_id()
    except Exception:  # noqa: BLE001 — manifest optional; strata degrade to indexed_unkeyed.
        by_id = {}

    per_case: list[dict[str, Any]] = []
    agg: collections.Counter[str] = collections.Counter()
    total_rel = 0
    for case in cases:
        rel = sorted(case.relevant_doc_ids)
        strata: collections.Counter[str] = collections.Counter(
            _stratum(d, indexed, by_id) for d in rel
        )
        agg += strata
        total_rel += len(rel)
        recoverable = sum(strata[s] for s in RECOVERABLE)
        per_case.append(
            {
                "id": case.id,
                "n_relevant": len(rel),
                "recall_ceiling": round(recoverable / len(rel), 3) if rel else None,
                "strata": dict(strata),
                "missing": [d for d in rel if _stratum(d, indexed, by_id) == "not_indexed"],
            }
        )

    recoverable_total = sum(agg[s] for s in RECOVERABLE)
    ceiling = recoverable_total / total_rel if total_rel else 0.0
    return {
        "profile": profile,
        "track": track,
        "n_cases": len(cases),
        "total_relevant": total_rel,
        "relevant_by_stratum": dict(agg),
        "recall_ceiling": round(ceiling, 4),
        "per_case": per_case,
    }


@click.command("coverage")
@click.option("--embed-profile", required=True)
@click.option("--track", default="catalog", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--show-worst", type=int, default=8, show_default=True, help="List the N lowest-ceiling cases."
)
def coverage_cmd(embed_profile: str, track: str, as_json: bool, show_worst: int) -> None:
    """Decompose the gold-set recall ceiling by corpus coverage (no retrieval)."""
    payload = compute_coverage(embed_profile, track=track)
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return
    strata = payload["relevant_by_stratum"]
    click.echo(
        f"track={payload['track']} cases={payload['n_cases']} "
        f"relevant_docs={payload['total_relevant']}"
    )
    for name in (
        "full_text",
        "metadata_only",
        "indexed_unkeyed",
        "not_indexed",
        "unknown",
        "failed",
    ):
        if name in strata:
            frac = strata[name] / payload["total_relevant"] if payload["total_relevant"] else 0.0
            click.echo(f"  {name:16s} {strata[name]:5d}  ({frac:.0%})")
    click.echo(
        f"RECALL CEILING = {payload['recall_ceiling']:.3f} "
        f"(max achievable recall given corpus coverage; the gap to actual recall@k is the "
        f"ranking-improvable headroom)."
    )
    worst = sorted(
        (c for c in payload["per_case"] if c["recall_ceiling"] is not None),
        key=lambda c: c["recall_ceiling"],
    )[:show_worst]
    if worst:
        click.echo("Lowest-ceiling cases (corpus gaps to close first):")
        for c in worst:
            click.echo(
                f"  {c['recall_ceiling']:.2f}  {c['id']}  "
                f"({c['n_relevant']} rel, {len(c['missing'])} not indexed)"
            )
