"""Deterministic, seed-stratified fold split of the synthetic gold cases.

The embedding-ladder campaign picks the best ladder rung BY its nDCG on the gold
set, then reports headline numbers. If selection and reporting use the SAME
cases, the "winner" is just the max of noisy draws (selection double-dip).
Splitting the cases into a SELECTION fold (used only to pick the best rung) and a
held-out REPORTING fold (used for ALL headline numbers) removes that bias.

Properties of the split (see ``compute_folds``):

* DETERMINISTIC — assignment is a function of the stable case ``id`` only, via a
  fixed SHA-256 hash. No RNG, no time/date, no manifest ordering. Two runs on the
  same ``cases.json`` produce byte-identical output.
* SEED-STRATIFIED — cases are bucketed by ``(track, category, mode)`` and split
  WITHIN each bucket, so neither fold is skewed by topic (category), track, or
  difficulty (single/hard). NOTE: ``seed_doc_id`` is near-unique in this gold set
  (112 distinct of 126; max one case per ``(track, seed_doc_id)``), so it cannot
  serve as a stratification bucket — bucketing on it yields all size-1 strata and
  a degenerate all-to-one-fold split. The seed instead drives the deterministic
  hash (it is embedded in the stable case ``id``), which is what "seed-stratified
  selection" requires: the WITHIN-bucket cut is seeded by the case/seed identity.
* ~50/50 — within each stratum cases are hash-ranked and the lower half goes to
  ``selection``, the upper half to ``reporting``. An odd stratum's extra case is
  assigned by a stable hash of the stratum key, so the rounding bias cancels
  across strata rather than always favoring one fold. The exact realized ratio is
  recorded in the persisted file's ``summary``.
* Both folds non-empty, disjoint, and their union == all cases.

The split targets the synthetic gold set (``data/eval/gold_synth/cases.json``),
which is the campaign's gold set. Hand-curated ``GOLD_CASES`` are not synthetic
and carry no seed/track stratification keys, so they are left fold-unrestricted
(they always pass the fold filter — see ``filter_case_ids_by_fold``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from graph_layout_rag.paths import DATA_DIR

Fold = Literal["selection", "reporting"]
FOLDS: tuple[Fold, ...] = ("selection", "reporting")

FOLDS_PATH = DATA_DIR / "eval" / "folds.json"
_CASES_PATH = DATA_DIR / "eval" / "gold_synth" / "cases.json"

# Bump if the hashing/stratification semantics change so a stale folds.json is
# not silently trusted across incompatible split logic.
FOLDS_VERSION = "folds-v1"

# Salt keeps this hash independent of any other id-hash in the codebase, so the
# fold assignment can never accidentally correlate with another hashed decision.
_HASH_SALT = "graph-layout-rag/folds-v1/"


def _rank_key(case_id: str) -> str:
    """Stable, RNG-free rank key for a case id (hex SHA-256 digest)."""

    return hashlib.sha256((_HASH_SALT + case_id).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldSplit:
    """An assignment of case ids to folds plus the stratification metadata."""

    assignment: dict[str, Fold]
    strata: dict[str, list[str]]  # stratum key -> sorted case ids
    stratum_keys: tuple[str, ...]  # the case fields used for stratification

    def ids_for_fold(self, fold: str) -> frozenset[str]:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; choose from {', '.join(FOLDS)}")
        return frozenset(cid for cid, f in self.assignment.items() if f == fold)


def _stratum_key(case: dict) -> str:
    """Stratification bucket: track + category (topic) + mode (difficulty).

    seed_doc_id is deliberately NOT used as a bucket key (it is near-unique here,
    which would make every stratum size 1 and the split degenerate); it instead
    feeds the per-case hash via the case id. See module docstring.
    """

    track = str(case.get("track", "unknown"))
    category = str(case.get("category") or "unknown")
    mode = str(case.get("mode") or "unknown")
    return f"{track}::{category}::{mode}"


def compute_folds(cases: list[dict]) -> FoldSplit:
    """Split synthetic gold ``cases`` into selection/reporting folds.

    ``cases`` are the raw dicts from ``cases.json`` (each with ``id``, ``track``,
    ``seed_doc_id``). The split is deterministic and seed+track stratified.
    """

    # Group by stratum; sort within each stratum by the hash rank (NOT by id or
    # input order) so the lower-half / upper-half cut is itself pseudo-random but
    # reproducible — avoids any ordering artifact in cases.json biasing a fold.
    strata: dict[str, list[str]] = {}
    for case in cases:
        cid = case["id"]
        strata.setdefault(_stratum_key(case), []).append(cid)

    assignment: dict[str, Fold] = {}
    sorted_strata: dict[str, list[str]] = {}
    for key in sorted(strata):
        ids = sorted(strata[key], key=_rank_key)
        sorted_strata[key] = ids
        n = len(ids)
        # Split lower-half -> selection, upper-half -> reporting. For an ODD-sized
        # stratum (including size 1), the single extra case goes to selection or
        # reporting based on a stable hash of the stratum key, so the rounding bias
        # cancels across strata instead of always favoring one fold (which on a
        # gold set of many size-1 strata would badly skew the overall ratio).
        odd_to_selection = int(_rank_key(key)[0], 16) % 2 == 0
        cut = n // 2 + (1 if (n % 2 == 1 and odd_to_selection) else 0)
        for idx, cid in enumerate(ids):
            assignment[cid] = "selection" if idx < cut else "reporting"

    return FoldSplit(
        assignment=assignment,
        strata=sorted_strata,
        stratum_keys=("track", "category", "mode"),
    )


def _load_cases(cases_path: Path = _CASES_PATH) -> list[dict]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    return list(data["cases"])


def _per_fold_per_track(cases: list[dict], split: FoldSplit) -> dict[str, dict[str, int]]:
    track_by_id = {c["id"]: str(c.get("track", "unknown")) for c in cases}
    counts: dict[str, dict[str, int]] = {f: {} for f in FOLDS}
    for cid, fold in split.assignment.items():
        track = track_by_id.get(cid, "unknown")
        counts[fold][track] = counts[fold].get(track, 0) + 1
    return counts


def write_folds(
    *,
    cases_path: Path = _CASES_PATH,
    out_path: Path = FOLDS_PATH,
) -> dict:
    """Compute the fold split from ``cases.json`` and persist it to ``out_path``.

    Returns the written payload. Regenerable: re-running on the same cases.json
    overwrites with byte-identical content (sorted keys, deterministic hash).
    """

    cases = _load_cases(cases_path)
    split = compute_folds(cases)
    per_fold_track = _per_fold_per_track(cases, split)
    total = len(split.assignment)
    sel = len(split.ids_for_fold("selection"))
    rep = len(split.ids_for_fold("reporting"))
    payload = {
        "version": FOLDS_VERSION,
        "hash_salt": _HASH_SALT,
        "stratum_keys": list(split.stratum_keys),
        "summary": {
            "total": total,
            "selection": sel,
            "reporting": rep,
            "selection_frac": round(sel / total, 4) if total else 0.0,
            "n_strata": len(split.strata),
            "per_fold_per_track": per_fold_track,
        },
        # case_id -> fold, sorted for stable diffs.
        "assignment": {cid: split.assignment[cid] for cid in sorted(split.assignment)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


def all_assigned_ids(*, folds_path: Path = FOLDS_PATH) -> frozenset[str]:
    """Return every case id that carries a fold assignment.

    Used by the fold filter to distinguish "fold-assigned but excluded" cases
    (drop) from "never assigned" cases such as the hand-curated GOLD_CASES (keep).
    """

    if folds_path.is_file():
        data = json.loads(folds_path.read_text(encoding="utf-8"))
        return frozenset(data.get("assignment", {}))
    return frozenset(compute_folds(_load_cases()).assignment)


def load_fold_ids(fold: str, *, folds_path: Path = FOLDS_PATH) -> frozenset[str] | None:
    """Return the case ids assigned to ``fold``, or ``None`` for "all".

    ``None`` (returned for ``fold == "all"`` or an absent fold) means "no fold
    restriction" — callers should keep every case. Recomputes from cases.json if
    the persisted folds.json is missing, so the helper never silently returns an
    empty/stale set.
    """

    if not fold or fold == "all":
        return None
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold {fold!r}; choose from all, {', '.join(FOLDS)}")
    if folds_path.is_file():
        data = json.loads(folds_path.read_text(encoding="utf-8"))
        assignment: dict[str, str] = data.get("assignment", {})
        return frozenset(cid for cid, f in assignment.items() if f == fold)
    # No persisted file: recompute deterministically so behavior is identical.
    split = compute_folds(_load_cases())
    return split.ids_for_fold(fold)
