"""T8 — judge transport routed through ``rag_common.local_llm`` + HG7 gate.

Covers:
* ``_judge_one`` dispatch: every backend routes to ``local_llm.generate_text``.
* The model-aware judge cache key carries the *active backend's* model, so
  grades from different backends never collide.
* The HG7 validation gate returns PASS/FAIL on synthetic agreement inputs above
  and below the rho>=0.70 / kappa>=0.70 / bias<0.25 thresholds.
"""

from __future__ import annotations

import pytest
from graph_layout_rag.eval import judge as judge_mod
from graph_layout_rag.eval.judge import _hg7_decide, _judge_one, judge_model, judge_pool

# ---------------------------------------------------------------------------
# _judge_one dispatch
# ---------------------------------------------------------------------------


def _doc() -> dict:
    return {"title": "Network Simplex layering", "abstract": "We layer via simplex."}


def test_judge_one_routes_to_local_llm_when_ollama(monkeypatch):
    # Explicit opt-out: the judge now defaults to the direct-HTTP dual backend
    # (DeepSeek primary / opencode-go fallback); GRAPH_RAG_JUDGE_BACKEND=local
    # is required to route through rag_common.local_llm instead.
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "local")
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    captured: dict = {}

    def fake_generate_text(prompt, *, model, temperature, max_tokens):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["temperature"] = temperature
        return '{"grade": 3, "reason": "spot on"}'

    monkeypatch.setattr("rag_common.local_llm.generate_text", fake_generate_text)

    grade, reason, actual_model = _judge_one("layer assignment", _doc(), "llama3.1:8b")
    assert (grade, reason) == (3, "spot on")
    assert actual_model == "llama3.1:8b"
    # transport got the resolved model and deterministic temperature
    assert captured["model"] == "llama3.1:8b"
    assert captured["temperature"] == 0.0
    # prompt is the real source-blind rubric prompt
    assert "Network Simplex" in captured["prompt"]


# ---------------------------------------------------------------------------
# cache key carries the active backend's model
# ---------------------------------------------------------------------------


def test_judge_model_resolves_ollama_model(monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "local")
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "mistral-small:24b")
    assert judge_model() == "mistral-small:24b"


def test_cache_key_carries_ollama_model(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "local")
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "mistral-small:24b")

    cache_path = tmp_path / "judge_cache.json"
    monkeypatch.setattr(judge_mod, "CACHE_PATH", cache_path)

    monkeypatch.setattr(
        "rag_common.local_llm.generate_text",
        lambda prompt, *, model, temperature, max_tokens: '{"grade": 3, "reason": "ok"}',
    )

    pool = {
        "track": "catalog",
        "cases": {
            "c1": {"query": "q", "pooled": {"d1": {"title": "T", "abstract": "a", "curated": True}}}
        },
    }
    judge_pool(pool)

    import json as _json

    cache = _json.loads(cache_path.read_text(encoding="utf-8"))
    # key format is "{model}:{case_id}:{doc_id}" — model must be the ollama name.
    assert "mistral-small:24b:c1:d1" in cache


# ---------------------------------------------------------------------------
# HG7 gate decision
# ---------------------------------------------------------------------------


def test_hg7_decide_pass():
    assert _hg7_decide(0.80, 0.75, 0.10) is True
    # boundary: thresholds are inclusive for rho/kappa, exclusive for bias
    assert _hg7_decide(0.70, 0.70, 0.24) is True


def test_hg7_decide_fail_each_axis():
    assert _hg7_decide(0.69, 0.90, 0.0) is False  # rho below
    assert _hg7_decide(0.90, 0.69, 0.0) is False  # kappa below
    assert _hg7_decide(0.90, 0.90, 0.25) is False  # bias at/over cap


class _FakeAudit:
    def __init__(self, audit: dict):
        self._audit = audit

    def judge_audit(self, *, pool_path, qrels_path, seed_cases_path=None):
        return self._audit


def test_validate_judge_hg7_pass(monkeypatch, tmp_path):
    """End-to-end HG7 with judge + audit mocked → PASS."""
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")

    pool_path = tmp_path / "pool.json"
    pool_path.write_text('{"track": "catalog", "cases": {}}', encoding="utf-8")

    # Stub the judge so we don't hit any LLM, and the audit loader.
    monkeypatch.setattr(judge_mod, "judge_pool", lambda pool, *, model=None: {})
    monkeypatch.setattr(
        judge_mod,
        "_load_analyze_bakeoff",
        lambda: _FakeAudit(
            {
                "seed_spearman": 0.82,
                "seed_kappa": 0.78,
                # Judge↔truth disagreement is similar across provenance → no bias.
                "dense_disagree_rate": 0.20,
                "bm25_disagree_rate": 0.10,
            }
        ),
    )

    result = judge_mod.validate_judge_hg7(pool_path)
    assert result["passed"] is True
    assert result["gate"] == "PASS"
    assert result["spearman"] == 0.82
    assert result["bias"] == pytest.approx(0.10)


def test_validate_judge_hg7_fail_bias(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    pool_path = tmp_path / "pool.json"
    pool_path.write_text('{"track": "catalog", "cases": {}}', encoding="utf-8")
    monkeypatch.setattr(judge_mod, "judge_pool", lambda pool, *, model=None: {})
    monkeypatch.setattr(
        judge_mod,
        "_load_analyze_bakeoff",
        lambda: _FakeAudit(
            {
                "seed_spearman": 0.90,
                "seed_kappa": 0.90,
                # Judge disagrees with curated truth far more for dense-surfaced
                # docs than bm25-surfaced ones → provenance bias, gap 0.40 > 0.25.
                "dense_disagree_rate": 0.70,
                "bm25_disagree_rate": 0.30,
            }
        ),
    )

    result = judge_mod.validate_judge_hg7(pool_path)
    assert result["passed"] is False
    assert result["gate"] == "FAIL"
    assert result["bias"] == pytest.approx(0.40)
