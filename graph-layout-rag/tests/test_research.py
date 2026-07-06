"""Offline unit tests for the `research` deep-research pipeline (mocked LLM + retrieval)."""

from __future__ import annotations

from typing import Any

import pytest
from graph_layout_rag.query import research as R


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _doc(doc_id: str, score: float = 0.9) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "canonical_doc_id": doc_id,
        "title": f"Title {doc_id}",
        "score": score,
        "excerpt": f"excerpt for {doc_id}",
        "evidence": [{"excerpt": f"excerpt for {doc_id}", "page": 1, "section_path": "s"}],
    }


class FakeGenerate:
    """Records every (step) call and returns canned per-step text."""

    def __init__(self, *, sufficient: str = "yes") -> None:
        self.sufficient = sufficient
        self.steps: list[str] = []

    def __call__(self, prompt: str, *, step: str, max_tokens: int, budget: R.Budget) -> str:
        self.steps.append(step)
        if step == "decompose":
            text = "1. dense retrieval basics\n- reranking effect\n\n3) chunking tradeoffs"
        elif step == "followup":
            text = "contextual retrieval failure modes"
        elif step == "evidence":
            text = "This paper contributes X to the question."
        elif step == "synthesis":
            text = f"Synthesis referencing [titlea2020].\nSUFFICIENT: {self.sufficient}"
        else:
            text = "ok"
        budget.record(step, prompt, text)
        return text


def _patch_pipeline(monkeypatch, gen: FakeGenerate, docs: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(R, "generate", gen)
    monkeypatch.setattr(R, "search", lambda q, **kw: [dict(d) for d in docs])
    monkeypatch.setattr(R, "expand_citations", lambda *a, **k: [])
    monkeypatch.setattr(R, "_manifest_index", lambda: {})


# --------------------------------------------------------------------------- #
# (1) decomposition parsing
# --------------------------------------------------------------------------- #
def test_decompose_parses_and_caps(monkeypatch):
    gen = FakeGenerate()
    monkeypatch.setattr(R, "generate", gen)
    budget = R.Budget()
    subs = R.decompose("What changed?", budget=budget, max_sub=4)
    assert subs == ["dense retrieval basics", "reranking effect", "chunking tradeoffs"]
    assert budget.calls == 1

    monkeypatch.setattr(R, "generate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert R.decompose("x", budget=R.Budget()) == []


# --------------------------------------------------------------------------- #
# (2) citation-expansion inclusion logic
# --------------------------------------------------------------------------- #
def test_select_neighbors_threshold_dedup_and_cap():
    present: set[str | None] = {"seed"}
    scores = {"a": 0.5, "b": 0.1, "c": 0.9, "seed": 0.99, "d": 0.4}

    def similarity(doc_id: str):
        if doc_id not in scores:
            return None
        return {
            "row": {"text": "t", "title": doc_id, "doc_id": doc_id, "page": 2},
            "score": scores[doc_id],
        }

    added = R._select_neighbors(
        ["a", "b", "seed", "c", "d", "missing"],
        present,
        canonicalize=lambda d: d,
        similarity=similarity,
        threshold=0.3,
        cap=2,
    )
    # b (0.1) below threshold; seed already present; missing has no similarity.
    # Kept in order a, c until cap=2 → d never reached.
    assert [x["doc_id"] for x in added] == ["a", "c"]
    assert all(x["via_citation"] for x in added)
    assert added[0]["evidence"][0]["score"] == 0.5


# --------------------------------------------------------------------------- #
# (3) dry-run purity — zero LLM calls
# --------------------------------------------------------------------------- #
def test_dry_run_makes_zero_llm_calls(monkeypatch):
    gen = FakeGenerate()
    _patch_pipeline(monkeypatch, gen, [_doc("d1"), _doc("d2")])

    result = R.research("q?", dry_run=True, max_docs=5)

    assert gen.steps == []  # generate never invoked
    assert result["cost"]["llm_calls"] == 0
    assert result["synthesis"] is None
    assert result["sufficient"] is None
    assert result["iterations"] == 0
    assert result["subquestions"] == ["q?"]  # no decompose on dry-run
    assert len(result["evidence"]) == 2


# --------------------------------------------------------------------------- #
# (4) depth cap — at most one iterate round
# --------------------------------------------------------------------------- #
def test_depth_cap_one_iterate(monkeypatch):
    gen = FakeGenerate(sufficient="no")  # always reports insufficient
    _patch_pipeline(monkeypatch, gen, [_doc("d1"), _doc("d2")])

    result = R.research("q?", depth=2, max_docs=8)

    # Exactly two synthesis calls (initial + one iterate), then stop despite "no".
    assert gen.steps.count("synthesis") == 2
    assert gen.steps.count("followup") == 1
    assert result["iterations"] == 2
    assert result["sufficient"] is False


def test_depth_one_no_iterate(monkeypatch):
    gen = FakeGenerate(sufficient="no")
    _patch_pipeline(monkeypatch, gen, [_doc("d1")])

    result = R.research("q?", depth=1, max_docs=8)

    assert gen.steps.count("synthesis") == 1
    assert "followup" not in gen.steps
    assert result["iterations"] == 1


# --------------------------------------------------------------------------- #
# (5) JSON schema shape
# --------------------------------------------------------------------------- #
def test_result_schema_shape(monkeypatch):
    gen = FakeGenerate(sufficient="yes")
    _patch_pipeline(monkeypatch, gen, [_doc("d1"), _doc("d2")])

    result = R.research("q?", depth=1, max_docs=8)

    for key in (
        "question",
        "strategy",
        "subquestions",
        "evidence",
        "references",
        "synthesis",
        "sufficient",
        "iterations",
        "cost",
        "dry_run",
    ):
        assert key in result, key
    assert result["strategy"] == "hybrid"
    assert result["subquestions"][0] == "q?"
    ev = result["evidence"][0]
    assert {
        "doc_id",
        "canonical_doc_id",
        "title",
        "score",
        "via_citation",
        "excerpts",
        "summary",
    } <= set(ev)
    assert result["references"][0]["bib_key"]  # non-empty synthesized key
    # cost accounting present and internally consistent
    cost = result["cost"]
    assert cost["llm_calls"] == len(gen.steps)
    assert cost["usd_est"] >= 0

    # markdown rendering is total and mentions the synthesis + references headers
    md = R.render_markdown(result)
    assert "## Synthesis" in md and "## References" in md and "## Evidence" in md


def test_bib_key_synthesis():
    assert (
        R._bib_key(["Karpukhin, Vladimir"], 2020, "Dense Passage Retrieval", "x")
        == "karpukhin2020dense"
    )
    assert R._bib_key([], None, None, "arxiv-1234") == "arxiv1234"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
