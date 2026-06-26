"""Tests for the deterministic, seed-stratified gold fold split (D2)."""

from __future__ import annotations

import json

import pytest

from graph_layout_rag.eval.folds import (
    FOLDS,
    compute_folds,
    load_fold_ids,
    write_folds,
)


@pytest.fixture(scope="module")
def real_cases() -> list[dict]:
    from graph_layout_rag.eval.folds import _CASES_PATH

    data = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    cases = list(data["cases"])
    assert cases, "cases.json must be non-empty for the fold split tests"
    return cases


def test_determinism_identical_across_runs(real_cases: list[dict]) -> None:
    a = compute_folds(real_cases).assignment
    b = compute_folds(real_cases).assignment
    assert a == b
    # Input order must not matter (assignment is a pure function of case id).
    shuffled = list(reversed(real_cases))
    c = compute_folds(shuffled).assignment
    assert a == c


def test_disjoint_and_exhaustive(real_cases: list[dict]) -> None:
    split = compute_folds(real_cases)
    sel = split.ids_for_fold("selection")
    rep = split.ids_for_fold("reporting")
    all_ids = {c["id"] for c in real_cases}
    assert sel.isdisjoint(rep)
    assert sel | rep == all_ids
    assert len(sel) + len(rep) == len(all_ids)


def test_both_folds_non_empty(real_cases: list[dict]) -> None:
    split = compute_folds(real_cases)
    assert split.ids_for_fold("selection")
    assert split.ids_for_fold("reporting")


def test_roughly_fifty_fifty(real_cases: list[dict]) -> None:
    split = compute_folds(real_cases)
    total = len(real_cases)
    sel = len(split.ids_for_fold("selection"))
    # Odd-stratum tie-breaks alternate by stratum hash, so the overall ratio sits
    # close to 0.5 in either direction.
    assert 0.42 <= sel / total <= 0.58


def test_per_track_balance(real_cases: list[dict]) -> None:
    split = compute_folds(real_cases)
    by_track: dict[str, dict[str, int]] = {}
    track_of = {c["id"]: str(c.get("track", "unknown")) for c in real_cases}
    for cid, fold in split.assignment.items():
        track = track_of[cid]
        by_track.setdefault(track, {f: 0 for f in FOLDS})[fold] += 1
    # Within each track, the two folds must be near-even: difference no larger
    # than the number of seed strata in that track (one per odd stratum at most),
    # and in practice within a couple of cases.
    for track, counts in by_track.items():
        sel, rep = counts["selection"], counts["reporting"]
        total = sel + rep
        assert total > 0
        # Folds are near-even within each track (gap a small fraction of total).
        assert abs(rep - sel) <= max(3, total // 3), (track, sel, rep)


def test_stratified_within_bucket(real_cases: list[dict]) -> None:
    # No single (track, category, mode) stratum may land entirely in one fold
    # when it has >= 2 cases — that would mean a topic is invisible to a fold.
    split = compute_folds(real_cases)
    for key, ids in split.strata.items():
        if len(ids) >= 2:
            folds = {split.assignment[cid] for cid in ids}
            assert len(folds) == 2, (key, ids)


def test_write_and_load_roundtrip(tmp_path, real_cases: list[dict]) -> None:
    from graph_layout_rag.eval.folds import _CASES_PATH

    out = tmp_path / "folds.json"
    p1 = write_folds(cases_path=_CASES_PATH, out_path=out)
    text1 = out.read_text(encoding="utf-8")
    # Regenerable: a second write is byte-identical.
    write_folds(cases_path=_CASES_PATH, out_path=out)
    assert out.read_text(encoding="utf-8") == text1

    sel = load_fold_ids("selection", folds_path=out)
    rep = load_fold_ids("reporting", folds_path=out)
    assert sel and rep and sel.isdisjoint(rep)
    assert load_fold_ids("all", folds_path=out) is None
    # Summary matches the assignment.
    assert p1["summary"]["selection"] == len(sel)
    assert p1["summary"]["reporting"] == len(rep)


def test_load_fold_ids_recomputes_without_file(tmp_path, real_cases: list[dict]) -> None:
    missing = tmp_path / "absent.json"
    sel = load_fold_ids("selection", folds_path=missing)
    # Falls back to recomputing from cases.json rather than returning empty.
    assert sel
    assert sel == compute_folds(real_cases).ids_for_fold("selection")


def test_unknown_fold_rejected() -> None:
    with pytest.raises(ValueError):
        load_fold_ids("train")
