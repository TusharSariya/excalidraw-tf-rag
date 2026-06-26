from __future__ import annotations

from dataclasses import replace
from typing import Literal

from graph_layout_rag.eval.gold_cases import EvalCase, gold_cases
from graph_layout_rag.manifest import Manifest, load_manifest

EvalTrack = Literal["catalog", "pdf-deep-read"]
EVAL_TRACKS: tuple[EvalTrack, ...] = ("catalog", "pdf-deep-read")


def validate_gold(manifest: Manifest | None = None) -> dict:
    manifest = manifest or load_manifest()
    by_id = {item.id: item for item in manifest.items}
    relevant_ids = sorted({doc_id for case in gold_cases() for doc_id in case.relevant_doc_ids})
    missing = [doc_id for doc_id in relevant_ids if doc_id not in by_id]
    metadata_only = [
        doc_id
        for doc_id in relevant_ids
        if doc_id in by_id and by_id[doc_id].status == "metadata_only"
    ]
    failed = [
        doc_id
        for doc_id in relevant_ids
        if doc_id in by_id and by_id[doc_id].status == "failed"
    ]
    impossible_pdf_cases = []
    for case in gold_cases():
        if not case.pdf_only:
            continue
        usable = [
            doc_id
            for doc_id in case.relevant_doc_ids
            if doc_id in by_id and by_id[doc_id].status == "ok" and by_id[doc_id].localPath
        ]
        if not usable:
            impossible_pdf_cases.append(case.id)
    return {
        "valid": not missing,
        "case_count": len(gold_cases()),
        "unique_relevant_doc_ids": len(relevant_ids),
        "missing_doc_ids": missing,
        "metadata_only_doc_ids": metadata_only,
        "failed_doc_ids": failed,
        "impossible_pdf_only_cases": impossible_pdf_cases,
    }


def _apply_fold_filter(cases: list[EvalCase]) -> list[EvalCase]:
    """Restrict cases to a fold when ``GRAPH_RAG_FOLD`` is set.

    Driven by an env var (like ``GRAPH_RAG_INCLUDE_SYNTH``/``GRAPH_RAG_QRELS_PATH``)
    so it propagates automatically to the isolated strategy subprocess workers.
    Cases whose id is not in the fold map (e.g. the hand-curated, non-synthetic
    ``GOLD_CASES``) are kept — only synthetic cases that ARE in the map and assigned
    to a different fold are dropped.
    """

    import os

    fold = os.getenv("GRAPH_RAG_FOLD")
    if not fold or fold == "all":
        return cases

    from graph_layout_rag.eval.folds import all_assigned_ids, load_fold_ids

    fold_ids = load_fold_ids(fold)
    if fold_ids is None:
        return cases
    # A case is dropped only if it is assigned to *some* fold but not the
    # requested one. Ids absent from the fold map (the hand-curated, non-synthetic
    # GOLD_CASES) pass through unrestricted.
    assigned = all_assigned_ids()
    return [c for c in cases if c.id not in assigned or c.id in fold_ids]


def cases_for_track(
    track: EvalTrack,
    *,
    manifest: Manifest | None = None,
) -> list[EvalCase]:
    if track not in EVAL_TRACKS:
        raise ValueError(f"Unknown eval track {track!r}; choose from {', '.join(EVAL_TRACKS)}")
    cases = _apply_fold_filter(gold_cases())
    if track == "catalog":
        return [replace(case, pdf_only=False) for case in cases]

    manifest = manifest or load_manifest()
    pdf_ids = {
        item.id
        for item in manifest.items
        if item.status == "ok" and item.localPath
    }
    out: list[EvalCase] = []
    for case in cases:
        relevant = frozenset(doc_id for doc_id in case.relevant_doc_ids if doc_id in pdf_ids)
        if not relevant:
            continue
        out.append(replace(case, relevant_doc_ids=relevant, pdf_only=True))
    return out
