"""CLI: multi-system pooling, LLM judging, and pooling-bias diagnostics.

Workflow::

    eval pool        --track catalog --embed-profile P [--splade-index D] [--colbert-index D]
    eval judge       --track catalog
    eval diagnostics --track catalog --embed-profile P --qrels data/eval/qrels/catalog/qrels.json
    eval benchmark   --embed-profile P --qrels data/eval/qrels/catalog/qrels.json --report
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from graph_layout_rag.eval.gold_validation import EVAL_TRACKS


def _experimental_indexes(splade_index: Path | None, colbert_index: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if splade_index:
        out["splade"] = str(splade_index)
    if colbert_index:
        out["colbert"] = str(colbert_index)
    return out


@click.command("pool")
@click.option("--track", type=click.Choice(EVAL_TRACKS), required=True)
@click.option("--embed-profile", required=True)
@click.option("--system", "systems", multiple=True, help="Pool system (repeatable); default diverse set.")
@click.option("--depth", default=50, show_default=True, type=click.IntRange(min=1))
@click.option("--splade-index", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--colbert-index", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--leave-dense-out",
    is_flag=True,
    help=(
        "Also emit a seed-system attribution sidecar (pool.attribution.json) and a "
        "leave-dense-out gold case-id list (pool.leave_dense_out.json): the cases "
        "whose synthetic seed was surfaced by a lexical/BM25-family system, so the "
        "BM25-falls comparison can be recomputed without dense/HyDE survivorship bias."
    ),
)
@click.option(
    "--qrels",
    "qrels_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional judged qrels for the leave-dense-out sidecar (fills seed grades).",
)
@click.option("-v", "--verbose", is_flag=True)
def pool_cmd(
    track: str,
    embed_profile: str,
    systems: tuple[str, ...],
    depth: int,
    splade_index: Path | None,
    colbert_index: Path | None,
    output: Path | None,
    leave_dense_out: bool,
    qrels_path: Path | None,
    verbose: bool,
) -> None:
    """Build a diverse multi-system candidate pool per gold case (de-biases the judge)."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    from graph_layout_rag.eval.pooling import build_pool, pool_path, pool_stats, write_pool

    payload = build_pool(
        track,  # type: ignore[arg-type]
        systems=list(systems) or None,
        depth=depth,
        embed_profile=embed_profile,
        experimental_indexes=_experimental_indexes(splade_index, colbert_index),
    )
    out_path = output or pool_path(track)  # type: ignore[arg-type]
    write_pool(payload, out_path)
    stats = pool_stats(payload)
    click.echo(f"Wrote {out_path}")
    click.echo(json.dumps({"systems": payload["systems"], **stats}, indent=2))

    if leave_dense_out:
        _write_leave_dense_out_sidecars(payload, out_path, qrels_path)


def _write_leave_dense_out_sidecars(
    payload: dict,
    pool_out_path: Path,
    qrels_path: Path | None,
) -> None:
    """Write per-case seed attribution + the leave-dense-out gold subset alongside a pool."""
    from graph_layout_rag.eval.gold_synth import load_synth_cases
    from graph_layout_rag.eval.pooling import (
        attribution_breakdown,
        leave_dense_out_case_ids,
        seed_attribution,
    )

    case_seed = {
        c["id"]: c["seed_doc_id"]
        for c in load_synth_cases()
        if c.get("seed_doc_id")
    }
    if not case_seed:
        click.echo("[leave-dense-out] no synthetic cases with seeds found; skipping sidecar.")
        return

    qrels = None
    if qrels_path is not None:
        from graph_layout_rag.eval.qrels import load_qrels

        qrels = load_qrels(qrels_path)

    attribution = seed_attribution(payload, case_seed, qrels=qrels)
    breakdown = attribution_breakdown(attribution)
    keep_ids = leave_dense_out_case_ids(attribution)

    attr_path = pool_out_path.with_name("pool.attribution.json")
    ldo_path = pool_out_path.with_name("pool.leave_dense_out.json")
    attr_path.write_text(
        json.dumps({"breakdown": breakdown, "cases": attribution}, indent=2) + "\n",
        encoding="utf-8",
    )
    ldo_path.write_text(
        json.dumps(
            {
                "description": (
                    "Gold case IDs whose synthetic seed was surfaced by a "
                    "lexical/BM25-family system (and judged relevant when grades "
                    "available); excludes dense/HyDE/neural-only survivors."
                ),
                "breakdown": breakdown,
                "case_ids": keep_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote {attr_path}")
    click.echo(f"Wrote {ldo_path}")
    click.echo(json.dumps(breakdown, indent=2))


@click.command("judge")
@click.option("--track", type=click.Choice(EVAL_TRACKS), default=None, help="Locate the default pool for this track.")
@click.option("--pool", "pool_file", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--model", default=None, help="Judge LLM model (default GRAPH_RAG_JUDGE_LLM_MODEL).")
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None)
@click.option("-v", "--verbose", is_flag=True)
def judge_cmd(
    track: str | None,
    pool_file: Path | None,
    model: str | None,
    output: Path | None,
    verbose: bool,
) -> None:
    """LLM-judge a pool (UMBRELA 0-3, source-blind) into a graded qrels file."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    from graph_layout_rag.eval.judge import judge_agreement, judge_model, judge_pool
    from graph_layout_rag.eval.pooling import pool_path
    from graph_layout_rag.eval.qrels import DEFAULT_RELEVANCE_THRESHOLD, write_qrels

    if pool_file is None:
        if track is None:
            raise click.ClickException("Pass --pool or --track to locate the pool.")
        pool_file = pool_path(track)  # type: ignore[arg-type]
        if not pool_file.is_file():
            raise click.ClickException(f"No pool at {pool_file}; run `eval pool --track {track}` first.")
    pool = json.loads(Path(pool_file).read_text(encoding="utf-8"))
    resolved_model = model or judge_model()
    cases_out = judge_pool(pool, model=resolved_model)

    track_name = pool.get("track", track or "catalog")
    from graph_layout_rag.paths import DATA_DIR

    out_path = output or (DATA_DIR / "eval" / "qrels" / track_name / "qrels.json")
    agreement = judge_agreement(cases_out)
    write_qrels(
        out_path,
        cases_out,
        judge_model=resolved_model,
        relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
        extra={"track": track_name, "pool": str(pool_file), "judge_agreement": agreement},
    )
    click.echo(f"Wrote {out_path}")
    click.echo(json.dumps(agreement, indent=2))


@click.command("judge-validate")
@click.option("--track", type=click.Choice(EVAL_TRACKS), default=None, help="Locate the default pool for this track.")
@click.option("--pool", "pool_file", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--model", default=None, help="Judge LLM model (default backend-resolved judge_model).")
@click.option("--seeds", "seeds_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("-v", "--verbose", is_flag=True)
def judge_validate_cmd(
    track: str | None,
    pool_file: Path | None,
    model: str | None,
    seeds_path: Path | None,
    verbose: bool,
) -> None:
    """HG7 judge-validation gate: agreement vs curated labels BEFORE qrels.

    PASS iff Spearman rho>=0.70 AND Cohen kappa>=0.70 AND dense-vs-BM25 bias<0.25.
    Run this to validate a (possibly local Ollama) judge before its grades feed
    the gold set. Exits non-zero on FAIL.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    from graph_layout_rag.eval.judge import validate_judge_hg7
    from graph_layout_rag.eval.pooling import pool_path

    if pool_file is None:
        if track is None:
            raise click.ClickException("Pass --pool or --track to locate the pool.")
        pool_file = pool_path(track)  # type: ignore[arg-type]
        if not pool_file.is_file():
            raise click.ClickException(f"No pool at {pool_file}; run `eval pool --track {track}` first.")

    result = validate_judge_hg7(pool_file, model=model, seed_cases_path=seeds_path)
    click.echo(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


@click.command("diagnostics")
@click.option("--track", type=click.Choice(EVAL_TRACKS), required=True)
@click.option("--embed-profile", required=True)
@click.option("--qrels", "qrels_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--strategy", "strategies", multiple=True, help="Strategy (repeatable); default bm25,dense,hybrid.")
@click.option("--depth", default=20, show_default=True, type=click.IntRange(min=1))
@click.option("--splade-index", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--colbert-index", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None)
def diagnostics_cmd(
    track: str,
    embed_profile: str,
    qrels_path: Path,
    strategies: tuple[str, ...],
    depth: int,
    splade_index: Path | None,
    colbert_index: Path | None,
    output: Path | None,
) -> None:
    """Hole rate / judged@k / condensed nDCG / bpref on old vs new qrels."""
    from graph_layout_rag.eval.diagnostics import format_diagnostics_table, run_diagnostics
    from graph_layout_rag.eval.qrels import load_qrels

    selected = list(strategies) or ["bm25", "dense", "hybrid"]
    payload = run_diagnostics(
        track=track,  # type: ignore[arg-type]
        embed_profile=embed_profile,
        qrels_payload=load_qrels(qrels_path),
        strategies=selected,
        depth=depth,
        experimental_indexes=_experimental_indexes(splade_index, colbert_index),
    )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        Path(output).with_suffix(".md").write_text(format_diagnostics_table(payload), encoding="utf-8")
    click.echo(format_diagnostics_table(payload))


@click.command("gate")
@click.option("--baseline-benchmark", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--candidate-benchmark", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--baseline-strategy", required=True)
@click.option("--candidate-strategy", required=True)
@click.option("--track", "tracks", type=click.Choice(EVAL_TRACKS), multiple=True)
@click.option(
    "--baseline-diagnostics",
    "baseline_diagnostics_paths",
    type=(click.Choice(EVAL_TRACKS), click.Path(path_type=Path, exists=True)),
    multiple=True,
    help="TRACK PATH pair; repeatable. Optional eval-diagnostics JSON for bpref join.",
)
@click.option(
    "--candidate-diagnostics",
    "candidate_diagnostics_paths",
    type=(click.Choice(EVAL_TRACKS), click.Path(path_type=Path, exists=True)),
    multiple=True,
    help="TRACK PATH pair; repeatable. Optional eval-diagnostics JSON for bpref join.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON to stdout.")
def gate_cmd(
    baseline_benchmark: Path,
    candidate_benchmark: Path,
    baseline_strategy: str,
    candidate_strategy: str,
    tracks: tuple[str, ...],
    baseline_diagnostics_paths: tuple[tuple[str, Path], ...],
    candidate_diagnostics_paths: tuple[tuple[str, Path], ...],
    as_json: bool,
) -> None:
    """Apply graph's calibrated 2026-06-18 promotion gate to a candidate retrieval change."""
    from graph_layout_rag.eval.gate import evaluate_gate, format_gate_report, load_json

    selected_tracks = list(tracks) or list(EVAL_TRACKS)
    baseline_payload = load_json(baseline_benchmark)
    candidate_payload = load_json(candidate_benchmark)
    baseline_diagnostics = {track: load_json(path) for track, path in baseline_diagnostics_paths}
    candidate_diagnostics = {track: load_json(path) for track, path in candidate_diagnostics_paths}

    result = evaluate_gate(
        baseline_benchmark=baseline_payload,
        candidate_benchmark=candidate_payload,
        baseline_strategy=baseline_strategy,
        candidate_strategy=candidate_strategy,
        tracks=selected_tracks,  # type: ignore[arg-type]
        baseline_diagnostics=baseline_diagnostics or None,
        candidate_diagnostics=candidate_diagnostics or None,
    )
    if as_json:
        click.echo(
            json.dumps(
                {
                    "passed": result.passed,
                    "findings": [f.__dict__ for f in result.findings],
                },
                indent=2,
            )
        )
    else:
        click.echo(format_gate_report(result))
    if not result.passed:
        raise SystemExit(1)


@click.command("pool-merge")
@click.argument("pool_files", nargs=-1, required=True, type=click.Path())
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), required=True)
@click.option("-v", "--verbose", is_flag=True)
def pool_merge_cmd(
    pool_files: tuple[str, ...],
    output: Path,
    verbose: bool,
) -> None:
    """Merge multiple pool JSON files into a single pool for joint judging.

    Example::

        eval pool-merge data/eval/pool/catalog/pool.json data/eval/pool/pdf-deep-read/pool.json \\
            -o data/eval/pool/merged/pool.json
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    from graph_layout_rag.eval.pooling import merge_pools, write_pool

    try:
        payload = merge_pools(list(pool_files))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    write_pool(payload, output)
    n_cases = payload["case_count"]
    n_candidates = sum(
        len(c.get("pooled", {})) for c in payload["cases"].values()
    )
    click.echo(f"Wrote {output}")
    click.echo(json.dumps({"n_cases": n_cases, "total_candidates": n_candidates}, indent=2))


@click.command("gen-gold")
@click.option("--embed-profile", required=True, help="Embed profile used to sample seed docs.")
@click.option("--n-catalog", default=90, show_default=True, type=int, help="Catalog seed count.")
@click.option("--n-pdf", default=40, show_default=True, type=int, help="PDF-deep-read seed count.")
@click.option("--hard-frac", default=0.30, show_default=True, type=float, help="Fraction of hard (indirect) queries.")
@click.option("--budget-usd", default=10.0, show_default=True, type=float, help="Approximate LLM cost budget.")
@click.option("--gen-only", is_flag=True, help="Generate and cache queries without writing cases.")
@click.option("-v", "--verbose", is_flag=True)
def gen_gold_cmd(
    embed_profile: str,
    n_catalog: int,
    n_pdf: int,
    hard_frac: float,
    budget_usd: float,
    gen_only: bool,
    verbose: bool,
) -> None:
    """Generate synthetic gold cases from manifest seeds (DataMorgana / ARES style).

    Reads manifest docs, samples stratified seeds, calls Gemini-Flash to write
    system-blind queries, applies quality filters (leakage, dup, short), and
    writes cases to data/eval/gold_synth/cases.json.

    After generation, run ``eval pool`` with GRAPH_RAG_INCLUDE_SYNTH=1 to pool
    the synthetic cases alongside the curated-49 gold set, then ``eval judge``
    to grade them.

    Requires GOOGLE_CLOUD_PROJECT / Vertex AI credentials for Gemini access.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    from graph_layout_rag.eval.gold_synth import build_synth_cases, write_cases

    click.echo(
        f"Generating synthetic gold cases: embed_profile={embed_profile} "
        f"n_catalog={n_catalog} n_pdf={n_pdf} hard_frac={hard_frac} "
        f"budget_usd={budget_usd}"
    )
    cases, manifest = build_synth_cases(
        embed_profile,
        n_catalog=n_catalog,
        n_pdf=n_pdf,
        hard_frac=hard_frac,
    )
    if not gen_only:
        write_cases(cases, manifest)
        click.echo(f"Wrote {len(cases)} cases to data/eval/gold_synth/cases.json")
    else:
        click.echo(f"Generated {len(cases)} cases (--gen-only: not written to disk)")
    click.echo(json.dumps(manifest, indent=2))


@click.command("corpus-health")
@click.option("--embed-profile", required=True)
@click.option("--track", type=click.Choice(EVAL_TRACKS), default="catalog", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit the raw audit JSON.")
def corpus_health_cmd(embed_profile: str, track: str, as_json: bool) -> None:
    """Audit corpus/infra health (chunk density, extraction fallback, creds, pool holes)."""
    from graph_layout_rag.eval.corpus_health import run_audit

    report = run_audit(embed_profile, track=track)
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    icon = {"critical": "x", "warning": "!", "info": ".", "ok": "ok"}
    click.echo(
        f"Corpus health - profile={report['profile']} track={report['track']} "
        f"worst={report['worst_severity']} "
        f"(critical={report['n_critical']} warning={report['n_warning']})\n"
    )
    for f in report["findings"]:
        click.echo(f"  {icon.get(f['severity'], '?')} [{f['severity']}] {f['code']}: {f['message']}")
        if f["remedy"]:
            click.echo(f"      -> {f['remedy']}")
    raise SystemExit(1 if report["n_critical"] else 0)
