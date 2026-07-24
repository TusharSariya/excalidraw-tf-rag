"""Unit tests for ArxivSource error propagation.

429s are deliberately *not* retried in-process: the worker leaves the batch
unarchived and pgmq redelivers it after VT_SECONDS. See ArxivSource.get_papers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import arxiv
import pytest
from harvest.sources.arxiv import ArxivSource


def _raising_source(status: int) -> ArxivSource:
    source = ArxivSource()
    source._client = MagicMock()
    err = arxiv.HTTPError(url="https://export.arxiv.org/api/query", retry=3, status=status)

    def boom_iter():
        raise err
        yield  # pragma: no cover

    source._client.results.return_value = boom_iter()
    return source


@pytest.mark.parametrize("status", [429, 500])
def test_get_papers_propagates_http_errors(status: int) -> None:
    source = _raising_source(status)

    with pytest.raises(arxiv.HTTPError) as ei:
        source.get_papers(["2106.15347"])

    assert ei.value.status == status


def test_get_papers_does_not_retry_429() -> None:
    """One request per call — no in-process backoff that could outrun the VT."""
    source = _raising_source(429)

    with pytest.raises(arxiv.HTTPError):
        source.get_papers(["2106.15347"])

    assert source._client.results.call_count == 1


def test_get_papers_empty_identifiers_makes_no_request() -> None:
    source = ArxivSource()
    source._client = MagicMock()

    assert source.get_papers([]) == {}
    source._client.results.assert_not_called()
