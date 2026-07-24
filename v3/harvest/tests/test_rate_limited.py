"""RateLimited makes the worker exit the drain instead of stampeding."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from harvest.errors import RateLimited
from harvest.sources.core import CoreSource


def test_core_429_raises_rate_limited() -> None:
    source = CoreSource(api_key="test-key")
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"x-ratelimit-retry-after": "2026-07-24T23:00:00+0000"}
    resp.text = "slow down"
    source._session = MagicMock()
    source._session.get.return_value = resp

    with pytest.raises(RateLimited) as ei:
        source.get_papers(["10.1111/cgf.12791"])

    assert ei.value.source == "core"
    assert ei.value.retry_after == "2026-07-24T23:00:00+0000"
    # One attempt only — no in-process retry storm.
    assert source._session.get.call_count == 1


def test_core_404_is_not_found_not_rate_limited() -> None:
    source = CoreSource(api_key="test-key")
    resp = MagicMock()
    resp.status_code = 404
    resp.headers = {}
    resp.text = "missing"
    source._session = MagicMock()
    source._session.get.return_value = resp

    assert source.get_papers(["10.1111/cgf.12791"]) == {"10.1111/cgf.12791": None}
