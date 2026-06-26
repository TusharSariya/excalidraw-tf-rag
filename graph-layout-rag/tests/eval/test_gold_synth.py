"""Tests for graph_layout_rag.eval.gold_synth."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from graph_layout_rag.eval.gold_synth import (
    LEAKAGE_JACCARD_MAX,
    DUP_JACCARD_MAX,
    FilterStats,
    Seed,
    _TAG_TO_CATEGORY,
    _jaccard,
    _tokens,
    category_for,
    load_synth_cases,
    quality_filters,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeItem:
    """Minimal stand-in for ManifestItem."""

    def __init__(self, tags: list[str], title: str = "") -> None:
        self.tags = tags
        self.title = title
        self.abstract = "Some abstract."
        self.year = 2020
        self.status = "ok"
        self.localPath = None
        self.id = "fake-id"


def _make_seed(
    doc_id: str = "seed-doc",
    title: str = "Crossing Minimization Survey",
    abstract: str = "We survey crossing minimization methods for layered graphs.",
    chunk_text: str = "",
) -> Seed:
    return Seed(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        chunk_text=chunk_text,
        category="crossing",
        year_bucket="2011-2020",
        track="catalog",
    )


# --------------------------------------------------------------------------- #
# _TAG_TO_CATEGORY: each graph-layout tag maps correctly
# --------------------------------------------------------------------------- #
_EXPECTED_TAG_MAPPINGS = {
    "layer-assignment": "layer-assignment",
    "layer": "layer-assignment",
    "crossing": "crossing",
    "compound": "compound",
    "coordinate-assignment": "coordinate-assignment",
    "coordinate": "coordinate-assignment",
    "constraints": "constraints",
    "constraint": "constraints",
    "routing": "routing",
    "ports": "ports",
    "port": "ports",
    "packing": "packing",
    "overlap": "overlap",
    "compaction": "compaction",
    "compact": "compaction",
    "elk": "elk",
}


@pytest.mark.parametrize("tag,expected_cat", list(_EXPECTED_TAG_MAPPINGS.items()))
def test_tag_to_category_known(tag: str, expected_cat: str) -> None:
    item = _FakeItem(tags=[tag])
    assert category_for(item) == expected_cat


def test_tag_to_category_unknown_fallback() -> None:
    """An unknown tag should fall back to the 'general' category."""
    item = _FakeItem(tags=["unknown-tag", "some-other-tag"])
    assert category_for(item) == "general"


def test_tag_to_category_empty_tags_fallback() -> None:
    item = _FakeItem(tags=[])
    assert category_for(item) == "general"


def test_tag_to_category_title_match() -> None:
    """Category should also be detectable from the title when tags are absent."""
    item = _FakeItem(tags=[], title="Efficient routing algorithms for orthogonal edges")
    assert category_for(item) == "routing"


# --------------------------------------------------------------------------- #
# quality_filters: leakage / dup / too-short
# --------------------------------------------------------------------------- #
def test_quality_filters_leakage() -> None:
    """A query that echoes the seed's own words should be rejected as leakage."""
    seed = _make_seed(
        title="Crossing Minimization Survey",
        abstract=(
            "We survey crossing minimization methods using barycenter and median "
            "heuristics for layered directed graphs with dummy nodes."
        ),
    )
    # Echo the seed verbatim — should trigger leakage
    leaky_query = (
        "crossing minimization methods barycenter median heuristics layered directed "
        "graphs dummy nodes survey"
    )
    seeds = [seed]
    modes = ["single"]
    queries = {0: leaky_query}

    kept, stats = quality_filters(seeds, modes, queries)
    assert stats.leakage >= 1
    assert len(kept) == 0


def test_quality_filters_near_dup_filtered() -> None:
    """Two queries that are near-duplicates of each other: the second should be dropped."""
    s1 = _make_seed(doc_id="doc-a", title="Paper A on Edge Bundling", abstract="Edge bundling hierarchical visualization method.")
    s2 = _make_seed(doc_id="doc-b", title="Paper B on Edge Bundling", abstract="Another edge bundling hierarchical visualization.")
    q1 = "how does hierarchical edge bundling reduce visual clutter in large graphs"
    q2 = "how does hierarchical edge bundling reduce visual clutter in large graphs"
    seeds = [s1, s2]
    modes = ["single", "single"]
    queries = {0: q1, 1: q2}

    kept, stats = quality_filters(seeds, modes, queries)
    assert stats.duplicate >= 1
    assert len(kept) == 1


def test_quality_filters_too_short() -> None:
    """A query with fewer than 4 words should be dropped."""
    seed = _make_seed()
    seeds = [seed]
    modes = ["single"]
    queries = {0: "crossing min"}

    kept, stats = quality_filters(seeds, modes, queries)
    assert stats.too_short >= 1
    assert len(kept) == 0


def test_quality_filters_passes_clean_query() -> None:
    """A well-formed, non-leaking query should pass all filters."""
    seed = _make_seed(
        title="Network Simplex Algorithm",
        abstract="The network simplex algorithm efficiently solves min-cost flow problems.",
    )
    good_query = "how to assign layers to nodes in a directed acyclic graph efficiently"
    seeds = [seed]
    modes = ["single"]
    queries = {0: good_query}

    kept, stats = quality_filters(seeds, modes, queries)
    assert stats.leakage == 0
    assert stats.too_short == 0
    assert len(kept) == 1


def test_quality_filters_above_leakage_threshold() -> None:
    """Jaccard just above LEAKAGE_JACCARD_MAX should be rejected."""
    # Build a query and seed where jaccard is artificially controllable.
    common = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    # Seed has exactly those tokens
    seed = _make_seed(title=common, abstract=common)
    # Query shares most of the same tokens — should exceed threshold
    query = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    qt = _tokens(query)
    st = _tokens(common + " " + common)
    jac = _jaccard(qt, st)
    # Verify our assumption about the leakage jaccard before asserting on filters
    if jac > LEAKAGE_JACCARD_MAX:
        kept, stats = quality_filters([seed], ["single"], {0: query})
        assert stats.leakage >= 1


def test_quality_filters_jaccard_below_dup_threshold_kept() -> None:
    """Two clearly different, non-leaking queries should both be kept."""
    # Seeds use intentionally generic abstracts with non-overlapping vocabulary from the queries.
    s1 = _make_seed(
        doc_id="a",
        title="Fruchterman Reingold Spring",
        abstract="Aesthetic criteria for placing vertices in an open plane region.",
    )
    s2 = _make_seed(
        doc_id="b",
        title="Column Based Layouts",
        abstract="Method for reducing total height of a hierarchical diagram using column assignment.",
    )
    # Queries are on different topics and don't echo the seed abstracts
    q1 = "how to optimize attractive repulsive forces in spring embedder placement"
    q2 = "techniques for reducing excessive vertical space in swimlane diagrams"
    seeds = [s1, s2]
    modes = ["single", "single"]
    queries = {0: q1, 1: q2}

    kept, stats = quality_filters(seeds, modes, queries)
    assert len(kept) == 2
    assert stats.duplicate == 0


# --------------------------------------------------------------------------- #
# load_synth_cases: interface contract
# --------------------------------------------------------------------------- #
def test_load_synth_cases_empty_when_no_file(tmp_path: Path) -> None:
    """load_synth_cases returns [] when no cases file exists."""
    nonexistent = tmp_path / "no-such-dir" / "cases.json"
    with mock.patch("graph_layout_rag.eval.gold_synth.CASES_PATH", nonexistent):
        result = load_synth_cases()
    assert result == []


def test_load_synth_cases_returns_list_with_required_keys(tmp_path: Path) -> None:
    """load_synth_cases returns list[dict] with id, query, relevant_ids, source."""
    cases_file = tmp_path / "cases.json"
    payload = {
        "manifest": {"version": "synth-v1"},
        "cases": [
            {
                "id": "synth-catalog-single-0000",
                "query": "how to minimize crossings in layered digraphs",
                "relevant_ids": ["crossing-paper-1"],
                "source": "synth",
                "category": "crossing",
                "track": "catalog",
                "mode": "single",
                "seed_doc_id": "crossing-paper-1",
                "notes": "synthetic;mode=single;seed=crossing-paper-1",
            }
        ],
    }
    cases_file.write_text(json.dumps(payload), encoding="utf-8")

    with mock.patch("graph_layout_rag.eval.gold_synth.CASES_PATH", cases_file):
        result = load_synth_cases()

    assert isinstance(result, list)
    assert len(result) == 1
    case = result[0]
    for key in ("id", "query", "relevant_ids", "source"):
        assert key in case, f"Missing required key: {key!r}"
    assert case["source"] == "synth"
    assert isinstance(case["relevant_ids"], list)


def test_load_synth_cases_malformed_json(tmp_path: Path) -> None:
    """load_synth_cases gracefully returns [] on malformed JSON."""
    cases_file = tmp_path / "cases.json"
    cases_file.write_text("{ not valid json }", encoding="utf-8")

    with mock.patch("graph_layout_rag.eval.gold_synth.CASES_PATH", cases_file):
        result = load_synth_cases()
    assert result == []


# --------------------------------------------------------------------------- #
# REGRESSION: gold_cases() returns EXACTLY 49 cases without GRAPH_RAG_INCLUDE_SYNTH
# --------------------------------------------------------------------------- #
def test_gold_cases_baseline_count_without_synth() -> None:
    """gold_cases() must return exactly 49 curated cases when GRAPH_RAG_INCLUDE_SYNTH is unset."""
    # Ensure the env var is absent
    env = {k: v for k, v in os.environ.items() if k != "GRAPH_RAG_INCLUDE_SYNTH"}
    env.pop("GRAPH_RAG_QRELS_PATH", None)  # also strip qrels overlay
    with mock.patch.dict(os.environ, env, clear=True):
        from graph_layout_rag.eval.gold_cases import gold_cases

        cases = gold_cases()
    assert len(cases) == 49, (
        f"Expected exactly 49 curated gold cases, got {len(cases)}. "
        "Did you add a case to GOLD_CASES without updating this regression test?"
    )
