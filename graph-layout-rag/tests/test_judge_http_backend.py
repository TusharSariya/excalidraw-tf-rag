"""Dual-backend HTTP judge (opencode-go + DeepSeek) with credit-failover.

Pure-unit: every HTTP call is mocked, so these run offline in the CI lane.
"""

from __future__ import annotations

import urllib.error

import pytest
from graph_layout_rag.eval import judge


def test_backend_order():
    assert judge._backend_order("auto") == ["opencode", "deepseek"]
    assert judge._backend_order("opencode") == ["opencode"]
    assert judge._backend_order("deepseek") == ["deepseek"]
    assert judge._backend_order("") == []
    assert judge._backend_order("unknown") == []


def test_judge_model_honors_http_backend(monkeypatch):
    monkeypatch.delenv("GRAPH_RAG_JUDGE_HTTP_MODEL", raising=False)
    monkeypatch.delenv("GRAPH_RAG_DEEPSEEK_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "auto")
    assert judge.judge_model() == "opencode-go/deepseek-v4-flash"
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "deepseek")
    assert judge.judge_model() == "deepseek-chat"


def test_judge_one_http_fails_over_on_credit(monkeypatch):
    calls = []

    def fake_opencode(q, d, m):
        calls.append("opencode")
        raise judge._JudgeCreditError("402 insufficient balance")

    def fake_deepseek(q, d, m):
        calls.append("deepseek")
        return (3, "grounded")

    monkeypatch.setitem(judge._HTTP_JUDGES, "opencode", fake_opencode)
    monkeypatch.setitem(judge._HTTP_JUDGES, "deepseek", fake_deepseek)

    assert judge._judge_one_http("q", {"title": "t"}, "m", "auto") == (3, "grounded")
    assert calls == ["opencode", "deepseek"]  # opencode first, then failover


def test_judge_one_http_raises_when_all_backends_fail(monkeypatch):
    def boom(q, d, m):
        raise judge._JudgeCreditError("402")

    monkeypatch.setitem(judge._HTTP_JUDGES, "opencode", boom)
    monkeypatch.setitem(judge._HTTP_JUDGES, "deepseek", boom)
    with pytest.raises(judge._JudgeCreditError):
        judge._judge_one_http("q", {"title": "t"}, "m", "auto")


def test_http_chat_grade_credit_code_raises_crediterror(monkeypatch):
    def fake_urlopen(req, timeout=90):
        raise urllib.error.HTTPError(req.full_url, 402, "Insufficient Balance", {}, None)

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(judge._JudgeCreditError):
        judge._http_chat_grade("http://x", "key", "m", "prompt")


def test_http_chat_grade_parses_openai_shape(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"grade\\": 2, \\"reason\\": \\"related\\"}"}}]}'

    monkeypatch.setattr(judge.urllib.request, "urlopen", lambda req, timeout=90: _Resp())
    assert judge._http_chat_grade("http://x", "key", "m", "prompt") == (2, "related")


def test_judge_one_routes_to_http_when_backend_set(monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_JUDGE_BACKEND", "opencode")
    monkeypatch.setitem(judge._HTTP_JUDGES, "opencode", lambda q, d, m: (2, "r"))
    assert judge._judge_one("q", {"title": "t"}, "m") == (2, "r")
