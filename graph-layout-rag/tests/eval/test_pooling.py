"""Tests for graph_layout_rag.eval.pooling.merge_pools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_layout_rag.eval.pooling import merge_pools


# --------------------------------------------------------------------------- #
# Helpers to build minimal pool JSON fixtures
# --------------------------------------------------------------------------- #
def _pool_fixture(
    *,
    cases: dict | None = None,
    systems: list[str] | None = None,
    track: str = "catalog",
    depth: int = 50,
    embed_profile: str = "test-profile",
) -> dict:
    return {
        "version": 1,
        "generated_at": "2026-06-24T00:00:00Z",
        "track": track,
        "depth": depth,
        "embed_profile": embed_profile,
        "systems": systems or [],
        "case_count": len(cases or {}),
        "cases": cases or {},
    }


def _candidate(
    doc_id: str,
    systems: dict[str, int],
    curated: bool = False,
    title: str | None = None,
) -> dict:
    return {
        "canonical_doc_id": doc_id,
        "doc_id": doc_id,
        "title": title or doc_id,
        "excerpt": None,
        "source_url": None,
        "systems": systems,
        "curated": curated,
    }


def _write_pool(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# merge_pools([]) → empty payload
# --------------------------------------------------------------------------- #
def test_merge_pools_empty_list() -> None:
    result = merge_pools([])
    assert result["case_count"] == 0
    assert result["cases"] == {}
    assert result["systems"] == []


# --------------------------------------------------------------------------- #
# merge_pools([single]) → passthrough
# --------------------------------------------------------------------------- #
def test_merge_pools_single_path(tmp_path: Path) -> None:
    pool = _pool_fixture(
        systems=["bm25", "dense"],
        cases={
            "case-a": {
                "query": "crossing minimization layered graph",
                "category": "crossing",
                "pdf_only": False,
                "curated_relevant": ["doc-x"],
                "pooled": {
                    "doc-x": _candidate("doc-x", {"bm25": 1, "dense": 2}, curated=True),
                    "doc-y": _candidate("doc-y", {"dense": 3}),
                },
            }
        },
    )
    path = _write_pool(tmp_path, "pool.json", pool)
    result = merge_pools([str(path)])

    assert result["case_count"] == 1
    assert "case-a" in result["cases"]
    pooled = result["cases"]["case-a"]["pooled"]
    assert "doc-x" in pooled
    assert "doc-y" in pooled
    # Systems should be preserved
    assert pooled["doc-x"]["systems"]["bm25"] == 1
    assert pooled["doc-x"]["curated"] is True


# --------------------------------------------------------------------------- #
# merge_pools([a, b]) with disjoint cases → all cases present
# --------------------------------------------------------------------------- #
def test_merge_pools_disjoint_cases(tmp_path: Path) -> None:
    pool_a = _pool_fixture(
        systems=["bm25"],
        cases={
            "case-layer": {
                "query": "layer assignment network simplex",
                "category": "layer-assignment",
                "pdf_only": False,
                "curated_relevant": ["doc-layer"],
                "pooled": {
                    "doc-layer": _candidate("doc-layer", {"bm25": 1}),
                },
            }
        },
    )
    pool_b = _pool_fixture(
        systems=["dense"],
        cases={
            "case-crossing": {
                "query": "crossing minimization barycenter heuristic",
                "category": "crossing",
                "pdf_only": True,
                "curated_relevant": ["doc-crossing"],
                "pooled": {
                    "doc-crossing": _candidate("doc-crossing", {"dense": 2}),
                },
            }
        },
    )
    path_a = _write_pool(tmp_path, "pool_a.json", pool_a)
    path_b = _write_pool(tmp_path, "pool_b.json", pool_b)

    result = merge_pools([str(path_a), str(path_b)])

    assert result["case_count"] == 2
    assert "case-layer" in result["cases"]
    assert "case-crossing" in result["cases"]
    # Systems union
    assert "bm25" in result["systems"]
    assert "dense" in result["systems"]


# --------------------------------------------------------------------------- #
# merge_pools([a, b]) with overlapping candidates → provenance union is correct
# --------------------------------------------------------------------------- #
def test_merge_pools_overlapping_candidates_provenance_union(tmp_path: Path) -> None:
    """When the same doc appears in both pools for the same case, system maps union."""
    shared_case = {
        "query": "stress majorization graph layout",
        "category": "constraints",
        "pdf_only": False,
        "curated_relevant": ["doc-stress"],
        "pooled": {
            "doc-stress": _candidate("doc-stress", {"bm25": 1}),
            "doc-extra": _candidate("doc-extra", {"bm25": 5}),
        },
    }
    pool_a = _pool_fixture(systems=["bm25"], cases={"case-stress": shared_case})

    # Pool B also retrieves doc-stress (via dense, at rank 2) + a new doc
    shared_case_b = {
        "query": "stress majorization graph layout",
        "category": "constraints",
        "pdf_only": False,
        "curated_relevant": ["doc-stress"],
        "pooled": {
            "doc-stress": _candidate("doc-stress", {"dense": 2}),
            "doc-new": _candidate("doc-new", {"dense": 1}),
        },
    }
    pool_b = _pool_fixture(systems=["dense"], cases={"case-stress": shared_case_b})

    path_a = _write_pool(tmp_path, "pa.json", pool_a)
    path_b = _write_pool(tmp_path, "pb.json", pool_b)

    result = merge_pools([str(path_a), str(path_b)])

    pooled = result["cases"]["case-stress"]["pooled"]
    # doc-stress should have provenance from BOTH systems
    stress_entry = pooled["doc-stress"]
    assert "bm25" in stress_entry["systems"]
    assert "dense" in stress_entry["systems"]
    assert stress_entry["systems"]["bm25"] == 1
    assert stress_entry["systems"]["dense"] == 2

    # doc-extra (only in pool_a) should still be present
    assert "doc-extra" in pooled
    # doc-new (only in pool_b) should also be present
    assert "doc-new" in pooled


def test_merge_pools_best_rank_kept_for_same_system(tmp_path: Path) -> None:
    """When the same system appears in two pools for the same doc, keep the lowest rank."""
    case_a = {
        "query": "port constraint fixed order",
        "category": "ports",
        "pdf_only": False,
        "curated_relevant": [],
        "pooled": {
            "doc-port": _candidate("doc-port", {"bm25": 10}),
        },
    }
    case_b = {
        "query": "port constraint fixed order",
        "category": "ports",
        "pdf_only": False,
        "curated_relevant": [],
        "pooled": {
            "doc-port": _candidate("doc-port", {"bm25": 3}),
        },
    }
    pool_a = _pool_fixture(systems=["bm25"], cases={"case-port": case_a})
    pool_b = _pool_fixture(systems=["bm25"], cases={"case-port": case_b})

    path_a = _write_pool(tmp_path, "pa.json", pool_a)
    path_b = _write_pool(tmp_path, "pb.json", pool_b)

    result = merge_pools([str(path_a), str(path_b)])
    # Best (lowest) rank should win
    rank = result["cases"]["case-port"]["pooled"]["doc-port"]["systems"]["bm25"]
    assert rank == 3


def test_merge_pools_curated_flag_propagates(tmp_path: Path) -> None:
    """curated=True in any pool should propagate to the merged result."""
    case_a = {
        "query": "some query",
        "category": None,
        "pdf_only": False,
        "curated_relevant": [],
        "pooled": {
            "doc-x": _candidate("doc-x", {"bm25": 1}, curated=False),
        },
    }
    case_b = {
        "query": "some query",
        "category": None,
        "pdf_only": False,
        "curated_relevant": [],
        "pooled": {
            "doc-x": _candidate("doc-x", {"dense": 2}, curated=True),
        },
    }
    pool_a = _pool_fixture(systems=["bm25"], cases={"case-x": case_a})
    pool_b = _pool_fixture(systems=["dense"], cases={"case-x": case_b})

    path_a = _write_pool(tmp_path, "pa.json", pool_a)
    path_b = _write_pool(tmp_path, "pb.json", pool_b)

    result = merge_pools([str(path_a), str(path_b)])
    assert result["cases"]["case-x"]["pooled"]["doc-x"]["curated"] is True


# --------------------------------------------------------------------------- #
# merge_pools(["/nonexistent.json"]) → raises ValueError
# --------------------------------------------------------------------------- #
def test_merge_pools_nonexistent_path_raises() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        merge_pools(["/nonexistent/path/pool.json"])


def test_merge_pools_malformed_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ broken json :", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse"):
        merge_pools([str(bad_file)])


def test_merge_pools_missing_cases_key_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "missing_cases.json"
    bad_file.write_text(json.dumps({"version": 1, "track": "catalog"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        merge_pools([str(bad_file)])
