"""Unit tests for arXiv request-id remapping (no network)."""

from __future__ import annotations

from harvest.sources.arxiv import _bare_id, requested_id_for


class _FakeResult:
    def __init__(self, short_id: str) -> None:
        self._short_id = short_id

    def get_short_id(self) -> str:
        return self._short_id


def test_bare_id_strips_version() -> None:
    assert _bare_id("1706.03762v1") == "1706.03762"
    assert _bare_id("1706.03762") == "1706.03762"


def test_requested_id_for_matches_versioned_result_to_bare_request() -> None:
    result = _FakeResult("1706.03762v1")
    assert requested_id_for(result, {"1706.03762", "1234.5678"}) == "1706.03762"


def test_requested_id_for_matches_exact_short_id() -> None:
    result = _FakeResult("1706.03762v1")
    assert requested_id_for(result, {"1706.03762v1"}) == "1706.03762v1"


def test_requested_id_for_returns_none_when_unmatched() -> None:
    result = _FakeResult("1706.03762v1")
    assert requested_id_for(result, {"9999.99999"}) is None
