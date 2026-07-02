"""`eval map-gate` + `eval judge-calibrate` — E1 map-quality gate CLI (C2).

Thin wiring around `rag_common.evals` (survey gold-set builder, ablation harness, judge
calibration) — the same tool-agnostic-library-plus-thin-adapter pattern as `rag_common.
litmap`/`query/litmap_cli.py`. Mirrored in graph-layout-rag; fix the twin too.

`map-gate` builds the survey gold set from this tool's citation DB + manifest, runs the
hybrid/text-only/citation-only ablation over the champion doc vectors, and writes a
JSON+markdown report to `data/eval/runs/<ts>-map-gate/`. It runs end-to-end even with a
sparse/absent citation graph (text_only-only arm, gate BLOCKs with a clear reason) — the
numbers are provisional until `cite enrich` finishes and the graph is re-keyed.

`judge-calibrate` ships the harness against the one judge-gated case that's actually
available pre-E1-lineage: cluster-label sanity ("does this extractive/LLM label describe
its cluster's papers?"). Lineage/faithfulness calibration needs E1 lineage output + the
user and is out of scope here (plan note).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

_CLUSTER_LABEL_RUBRIC = (
    "You are sanity-checking automatically generated cluster labels for a RAG / neural-IR "
    "literature research corpus. Given a cluster's label and a sample of its member paper "
    "titles, answer whether the label plausibly describes the shared topic of those "
    "titles.\n\nAnswer with exactly one word: yes or no."
)


def _run_dir() -> Path:
    from rag_literature_rag.paths import DATA_DIR

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DATA_DIR / "eval" / "runs" / f"{ts}-map-gate"


@click.command("map-gate")
@click.option("--profile", default=None, help="Embed profile whose index supplies doc vectors.")
@click.option("--min-refs", default=100, show_default=True, help="Survey ref-count threshold.")
@click.option("--top-k", default=10, show_default=True, help="Neighbors per doc in the kNN graph.")
@click.option(
    "--n-random", default=100, show_default=True, help="Null-distribution draws per survey."
)
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--fallback-keywords",
    is_flag=True,
    help="Widen survey detection to handbook/monograph/chapter if <5 surveys qualify.",
)
@click.option(
    "--target-k",
    default=None,
    type=int,
    help="Granularity target (>=2-doc cluster count) every arm is resolution-swept to "
    "match, for a valid cross-arm comparison. Default: the hybrid arm's natural "
    "(resolution=1.0) >=2-doc cluster count.",
)
@click.option(
    "--alpha-sweep",
    is_flag=True,
    help="Also run alpha in {0.25, 0.75} (in addition to 0.0/0.5/1.0) at matched K and "
    "report the best-scoring interior alpha.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON payload to stdout.")
def map_gate_cmd(
    profile: str | None,
    min_refs: int,
    top_k: int,
    n_random: int,
    seed: int,
    fallback_keywords: bool,
    target_k: int | None,
    alpha_sweep: bool,
    as_json: bool,
) -> None:
    """E1 promotion gate: hybrid vs text-only vs citation-only clustering, scored by
    reference-concentration on the corpus's in-corpus surveys, at a granularity-matched
    cluster count (see `rag_common.evals.ablation` docstring for why raw cross-arm
    comparison at mismatched cluster counts is invalid)."""
    from rag_common.evals import survey_gold as sg
    from rag_common.evals.ablation import ALPHA_SWEEP_VARIANTS, render_markdown_report, run_ablation

    from rag_literature_rag import citation_store as cs
    from rag_literature_rag.manifest import load_manifest
    from rag_literature_rag.query.litmap_support import (
        citation_adjacency,
        load_or_build_doc_vectors,
    )

    manifest = load_manifest()
    doc_meta = {
        item.id: sg.DocMeta(item.id, item.title, tuple(item.tags)) for item in manifest.items
    }

    surveys: list[sg.SurveyGold] = []
    adj = None
    if cs.CITATIONS_DB_PATH.exists():
        db = cs.connect()
        try:
            surveys = sg.find_surveys(db, doc_meta, min_refs=min_refs)
            if fallback_keywords and len(surveys) < 5:
                surveys = sg.find_surveys(
                    db, doc_meta, min_refs=min_refs, use_fallback_keywords=True
                )
        finally:
            db.close()
        adj = citation_adjacency()
    else:
        click.echo(
            "No citation store yet (run: rag-literature-rag cite enrich) — map-gate will "
            "run text_only-only; numbers are provisional.",
            err=True,
        )

    gold = sg.to_gold_dict(surveys)

    try:
        vectors, paths = load_or_build_doc_vectors(profile)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    if not vectors:
        click.echo(f"Index for profile {paths.profile!r} has no chunks.", err=True)
        sys.exit(1)

    variants = ALPHA_SWEEP_VARIANTS if alpha_sweep else None
    payload = run_ablation(
        vectors,
        adj,
        gold,
        top_k=top_k,
        seed=seed,
        n_random=n_random,
        target_k=target_k,
        **({"variants": variants} if variants is not None else {}),
    )
    payload["embed_profile"] = paths.profile
    payload["qualifying_survey_count"] = len(surveys)
    payload["min_refs"] = min_refs
    payload["provisional"] = not cs.CITATIONS_DB_PATH.exists()
    payload["alpha_sweep"] = alpha_sweep

    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown_report(payload), encoding="utf-8")
    click.echo(f"Wrote {run_dir}/report.json + report.md", err=True)

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(
        f"Qualifying surveys: {payload['qualifying_survey_count']} "
        f"(min_refs={min_refs}{', + fallback keywords' if fallback_keywords else ''})"
    )
    if payload["provisional"]:
        click.echo("PROVISIONAL: no citation store yet.")
    click.echo(f"target_k (granularity-matched >=2-doc clusters): {payload.get('target_k')}")
    for name, arm in payload["arms"].items():
        z = arm.get("mean_z_size_matched")
        cov = arm.get("coverage")
        click.echo(
            f"  {name:14s} alpha={arm['alpha']:.2f} resolution={arm.get('resolution', 1.0):.3f} "
            f"clusters={arm['cluster_count']:4d} clusters(>=2)={arm.get('cluster_count_min2'):4} "
            f"coverage={'n/a' if cov is None else f'{cov:.3f}'} "
            f"z(size-matched)={'n/a' if z is None else f'{z:.3f}'}"
        )
    for name, pc in payload.get("paired", {}).items():
        md = pc.get("mean_diff")
        lo, hi = pc.get("ci_low"), pc.get("ci_high")
        click.echo(
            f"  paired vs {name:14s} mean_diff={'n/a' if md is None else f'{md:.4f}'} "
            f"CI={'n/a' if lo is None else f'[{lo:.4f}, {hi:.4f}]'} "
            f"(n={pc.get('n_surveys')})"
        )
    if payload.get("best_alpha"):
        click.echo(f"Best alpha variant: {payload['best_alpha']}")
    click.echo(f"VERDICT: {payload['gate']['gate']}")
    for reason in payload["gate"]["reasons"]:
        click.echo(f"  - {reason}")


def _cluster_label_cases(profile: str | None) -> list[dict[str, Any]]:
    """Build one judge case per cluster from `map clusters` (default alpha/seed — fixed
    seed makes this reproducible between `--build-sheet` and `--score` runs, as long as
    the doc-vector cache / citation graph haven't changed in between)."""
    from rag_common.litmap.communities import build_clusters
    from rag_common.litmap.similarity import bib_coupling_all, cosine_knn_all, fuse_neighbors

    from rag_literature_rag.query.litmap_support import (
        citation_adjacency,
        doc_meta_map,
        load_or_build_doc_vectors,
    )

    vectors, _paths = load_or_build_doc_vectors(profile)
    doc_ids = list(vectors)
    text_knn = cosine_knn_all(vectors, top_k=10)
    adj = citation_adjacency()
    cite_knn = bib_coupling_all(doc_ids, adj, top_k=10) if adj is not None else {}

    edge_scores: dict[tuple[str, str], float] = {}
    for doc_id in doc_ids:
        fused = fuse_neighbors(
            text_knn.get(doc_id, []), cite_knn.get(doc_id, []), alpha=0.5, top_k=10
        )
        for n in fused:
            key = (doc_id, n.doc_id) if doc_id < n.doc_id else (n.doc_id, doc_id)
            edge_scores[key] = max(edge_scores.get(key, 0.0), n.score)
    edges = [(u, v, w) for (u, v), w in edge_scores.items()]

    meta = doc_meta_map()
    result = build_clusters(edges, meta, all_doc_ids=doc_ids, seed=42)
    cases = []
    for cluster in result["clusters"]:
        if cluster["size"] < 2:
            continue
        titles = [meta.get(d, {}).get("title") or d for d in cluster["doc_ids"][:12]]
        content = f"Cluster label: {cluster['label']!r}\n\nMember paper titles:\n" + "\n".join(
            f"- {t}" for t in titles
        )
        cases.append({"id": f"cluster-{cluster['id']}", "content": content})
    return cases


def _judge_yes_no(case: dict[str, Any]) -> str:
    from rag_common.local_llm import generate_text

    from rag_literature_rag.eval.judge import judge_model

    prompt = f"{_CLUSTER_LABEL_RUBRIC}\n\n{case['content']}"
    text = generate_text(prompt, model=judge_model(), temperature=0.0, max_tokens=8)
    normalized = (text or "").strip().lower()
    return "yes" if normalized.startswith("y") else "no"


@click.command("judge-calibrate")
@click.option(
    "--build-sheet", is_flag=True, help="Write a labeling sheet for cluster-label sanity."
)
@click.option(
    "--score",
    "score_sheet_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Score a filled labeling sheet against the live judge.",
)
@click.option("--n", default=30, show_default=True, help="Cases to sample for the sheet.")
@click.option(
    "--profile", default=None, help="Embed profile for the underlying `map clusters` run."
)
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
def judge_calibrate_cmd(
    build_sheet: bool,
    score_sheet_path: Path | None,
    n: int,
    profile: str | None,
    output: Path | None,
) -> None:
    """Judge calibration harness (D7/decisions-log item 19): hand-label cluster-label
    sanity cases, then check the live LLM judge's kappa against those labels before
    trusting any judge-gated map metric.

    Lineage/faithfulness calibration needs E1 lineage output + the user and is not built
    here yet — this ships the cluster-label-sanity slice, which is available now."""
    from rag_common.evals.judge_calibration import (
        build_labeling_sheet,
        decide,
        parse_labeling_sheet,
        score_agreement,
    )

    from rag_literature_rag.paths import DATA_DIR

    if build_sheet and score_sheet_path:
        click.echo("Pass exactly one of --build-sheet or --score.", err=True)
        sys.exit(1)
    if not build_sheet and not score_sheet_path:
        click.echo(
            "Pass --build-sheet to create a sheet, or --score <path> to grade one.", err=True
        )
        sys.exit(1)

    calib_dir = DATA_DIR / "eval" / "judge_calibration"

    if build_sheet:
        cases = _cluster_label_cases(profile)
        if not cases:
            click.echo("No clusters with >=2 members to sample — build an index first.", err=True)
            sys.exit(1)
        out_path = output or calib_dir / "cluster_label_sheet.md"
        build_labeling_sheet(cases, out_path, rubric=_CLUSTER_LABEL_RUBRIC, n=n)
        click.echo(f"Wrote labeling sheet: {out_path}")
        click.echo("Hand-fill the Label: lines (yes/no), then re-run with --score <path>.")
        return

    assert score_sheet_path is not None
    labels = parse_labeling_sheet(score_sheet_path)
    if not labels:
        click.echo(f"No filled labels found in {score_sheet_path}.", err=True)
        sys.exit(1)
    cases = _cluster_label_cases(profile)
    result = score_agreement(labels, cases, _judge_yes_no)
    decision = decide(result)
    payload = {**result.to_dict(), "decision": decision}

    out_path = output or calib_dir / "cluster_label_agreement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    click.echo(f"kappa={result.kappa:.3f} (n={result.n_cases}, min={payload['min_kappa']})")
    click.echo(f"outcome: {decision['outcome']} — {decision['action']}")
    click.echo(f"Wrote {out_path}")
