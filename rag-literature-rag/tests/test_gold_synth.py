"""Unit tests for the synthetic-gold filter logic (the bias guards).

Covers the deterministic, no-LLM core: category mapping, anti-leakage rejection,
near-duplicate rejection (within-set and vs curated), and the cases.json round-trip.
"""

from __future__ import annotations

import json
from collections import defaultdict
from random import Random

from rag_literature_rag.eval.gold_cases import GOLD_CASES
from rag_literature_rag.eval.gold_synth import (
    MULTIHOP_STRATUM,
    Seed,
    append_synth_cases,
    build_multihop_cases,
    category_for,
    load_synth_cases,
    quality_filters,
    write_cases,
)
from rag_literature_rag.manifest import ManifestItem


def _seed(doc_id: str, title: str, abstract: str) -> Seed:
    return Seed(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        chunk_text="",
        category="dense-retrieval",
        year_bucket=">=2024",
        track="catalog",
    )


def test_category_for_maps_tags_to_curated_vocab():
    item = ManifestItem(
        id="x",
        title="A ColBERT late interaction study",
        source="s",
        url="u",
        status="metadata_only",
        tags=["reranking"],
    )
    assert category_for(item) in {"reranking", "dense-retrieval"}


def test_leakage_query_is_rejected():
    seed = _seed(
        "d1", "Dense Passage Retrieval for Open Domain QA", "bi-encoder dense passage retrieval"
    )
    # Query echoes the seed wording → high Jaccard → leakage drop.
    leaky = {0: "Dense passage retrieval for open domain question answering bi-encoder"}
    kept, stats = quality_filters([seed], ["single"], leaky)
    assert kept == []
    assert stats.leakage == 1


def test_independent_phrasing_survives():
    seed = _seed(
        "d1", "Dense Passage Retrieval for Open Domain QA", "bi-encoder dense passage retrieval"
    )
    good = {0: "how to embed questions and documents separately to find answers fast"}
    kept, stats = quality_filters([seed], ["single"], good)
    assert len(kept) == 1
    assert stats.kept == 1


def test_near_duplicate_within_set_is_dropped():
    s1 = _seed("d1", "Paper One", "topic about graph retrieval clustering")
    s2 = _seed("d2", "Paper Two", "different subject entirely about audio")
    q = "techniques for graph based community detection in knowledge retrieval"
    queries = {0: q, 1: q + " methods"}  # second is a near-duplicate
    kept, stats = quality_filters([s1, s2], ["single", "single"], queries)
    assert stats.kept == 1
    assert stats.duplicate == 1


def test_collision_with_curated_query_is_dropped():
    seed = _seed("d1", "Some Paper", "unrelated wording here")
    curated_q = GOLD_CASES[0].query
    kept, stats = quality_filters([seed], ["single"], {0: curated_q})
    # Either flagged as duplicate of curated or as leakage; must not survive.
    assert kept == []


def test_cases_json_roundtrip(tmp_path, monkeypatch):
    from rag_literature_rag.eval import gold_synth

    monkeypatch.setattr(gold_synth, "SYNTH_DIR", tmp_path)
    monkeypatch.setattr(gold_synth, "CASES_PATH", tmp_path / "cases.json")
    from rag_literature_rag.eval.gold_cases import EvalCase

    cases = [
        EvalCase(
            id="synth-catalog-single-0001",
            query="example synthetic query about retrieval evaluation",
            relevant_doc_ids=frozenset({"arxiv-2005-11401"}),
            category="evaluation",
            notes="synthetic;mode=single;seed=arxiv-2005-11401",
        )
    ]
    write_cases(cases, {"version": "synth-v1", "total_cases": 1})
    loaded = load_synth_cases()
    assert len(loaded) == 1
    assert loaded[0].id == "synth-catalog-single-0001"
    assert loaded[0].relevant_doc_ids == frozenset({"arxiv-2005-11401"})


# --------------------------------------------------------------------------- #
# Multi-hop / relational stratum
# --------------------------------------------------------------------------- #
class _FakeGraph:
    """Citation graph stub keyed like CitationGraph (oa-id space)."""

    def __init__(self, edges):
        self.doc_to_oa = {}
        self.oa_to_doc = {}
        self.undirected = defaultdict(set)
        for a, b in edges:
            for d in (a, b):
                self.doc_to_oa[d] = f"oa-{d}"
                self.oa_to_doc[f"oa-{d}"] = d
            self.undirected[f"oa-{a}"].add(f"oa-{b}")
            self.undirected[f"oa-{b}"].add(f"oa-{a}")


def _mh_seed(doc_id: str, category: str, words: str) -> Seed:
    return Seed(
        doc_id=doc_id,
        title=f"Paper {doc_id}",
        abstract=words,
        chunk_text="",
        category=category,
        year_bucket=">=2024",
        track="catalog",
    )


_GOOD_MH_QUERY = "how does one retriever design relate to and improve upon another approach"

_MH_VARIANTS = [
    "how do dense bi-encoders compare with late interaction scoring for passage ranking",
    "what distinguishes sparse lexical expansion from learned reranking in retrieval pipelines",
    "when does graph structured summarization outperform flat chunk retrieval for questions",
    "why choose query rewriting over document expansion when recall is limited overall",
]


def _cycling_gen(monkeypatch):
    counter = {"n": 0}

    def _gen(*_a, **_k):
        v = _MH_VARIANTS[counter["n"] % len(_MH_VARIANTS)]
        counter["n"] += 1
        return v

    monkeypatch.setattr("rag_common.local_llm.generate_text", _gen)


def test_build_multihop_co_category(monkeypatch, tmp_path):
    seeds = [
        _mh_seed("d1", "dense-retrieval", "biencoder alpha beta"),
        _mh_seed("d2", "dense-retrieval", "colbert gamma delta"),
        _mh_seed("d3", "dense-retrieval", "splade epsilon zeta"),
        _mh_seed("d4", "dense-retrieval", "dpr eta theta"),
    ]
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth.GEN_CACHE_PATH", tmp_path / "gc.json")
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth._gen_model", lambda: "test-model")
    _cycling_gen(monkeypatch)

    cases, manifest = build_multihop_cases("prof", 2, seeds=seeds, graph=None, rng=Random(0))
    assert len(cases) == 2
    for c in cases:
        assert c["stratum"] == MULTIHOP_STRATUM
        a, b = c["seed_doc_id"], c["partner_doc_id"]
        assert set(c["relevant_doc_ids"]) == {a, b}
        assert c["grades"][a] == 3 and c["grades"][b] == 2
        assert c["pairing"] == "co-category"


def test_build_multihop_citation_pairing(monkeypatch, tmp_path):
    seeds = [
        _mh_seed("a", "reranking", "cross encoder alpha"),
        _mh_seed("b", "graphrag", "community summary beta gamma"),
    ]
    graph = _FakeGraph([("a", "b")])
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth.GEN_CACHE_PATH", tmp_path / "gc.json")
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth._gen_model", lambda: "test-model")
    monkeypatch.setattr("rag_common.local_llm.generate_text", lambda *a, **k: _GOOD_MH_QUERY)

    cases, manifest = build_multihop_cases("prof", 1, seeds=seeds, graph=graph, rng=Random(0))
    assert len(cases) == 1
    assert cases[0]["pairing"] == "citation"
    assert manifest["citation_linked"] == 1
    assert set(cases[0]["relevant_doc_ids"]) == {"a", "b"}


def test_build_multihop_leakage_dropped(monkeypatch, tmp_path):
    seeds = [
        _mh_seed(
            "d1", "dense-retrieval", "biencoder dense passage retrieval contrastive negatives"
        ),
        _mh_seed("d2", "dense-retrieval", "late interaction token level maxsim scoring colbert"),
    ]
    leaky = (
        "biencoder dense passage retrieval contrastive negatives late interaction token "
        "level maxsim scoring colbert"
    )
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth.GEN_CACHE_PATH", tmp_path / "gc.json")
    monkeypatch.setattr("rag_literature_rag.eval.gold_synth._gen_model", lambda: "test-model")
    monkeypatch.setattr("rag_common.local_llm.generate_text", lambda *a, **k: leaky)

    cases, manifest = build_multihop_cases("prof", 1, seeds=seeds, graph=None, rng=Random(0))
    assert len(cases) == 0
    assert manifest["filters"]["leakage"] >= 1


def test_append_synth_cases_additive(tmp_path, monkeypatch):
    from rag_literature_rag.eval import gold_synth

    cases_file = tmp_path / "cases.json"
    existing = {
        "manifest": {"version": "synth-v1", "total_cases": 1},
        "cases": [
            {
                "id": "synth-catalog-single-0000",
                "query": "existing single-hop query about retrieval",
                "relevant_doc_ids": ["doc-x"],
                "pdf_only": False,
                "stratum": "single-hop",
            }
        ],
    }
    cases_file.write_text(json.dumps(existing), encoding="utf-8")
    monkeypatch.setattr(gold_synth, "SYNTH_DIR", tmp_path)
    monkeypatch.setattr(gold_synth, "CASES_PATH", cases_file)

    new = [
        {
            "id": "synth-catalog-multihop-0000",
            "query": "q1",
            "relevant_doc_ids": ["a", "b"],
            "stratum": MULTIHOP_STRATUM,
        },
        {
            "id": "synth-catalog-multihop-0001",
            "query": "q2",
            "relevant_doc_ids": ["c", "d"],
            "stratum": MULTIHOP_STRATUM,
        },
    ]
    summary = append_synth_cases(new)
    summary2 = append_synth_cases(new)
    loaded = load_synth_cases()

    assert summary == {"added": 2, "total": 3, "skipped_existing": 0}
    assert summary2["added"] == 0
    assert len(loaded) == 3
    ids = {c.id for c in loaded}
    assert "synth-catalog-single-0000" in ids
    mh = [c for c in loaded if len(c.relevant_doc_ids) == 2]
    assert len(mh) == 2
    payload = json.loads(cases_file.read_text(encoding="utf-8"))
    assert payload["manifest"]["by_stratum"][MULTIHOP_STRATUM] == 2
