"""Unit tests for ArxivSource outer 429 retry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import arxiv
import pytest
from harvest.sources.arxiv import ArxivSource


def test_get_papers_retries_429_then_succeeds() -> None:
    source = ArxivSource()
    source._client = MagicMock()

    boom = arxiv.HTTPError(url="https://export.arxiv.org/api/query", retry=3, status=429)
    result = MagicMock()
    result.get_short_id.return_value = "2106.15347"

    # First results() iterator raises on first next(); second yields one result.
    def results_side_effect(_search: arxiv.Search):
        if not hasattr(results_side_effect, "n"):
            results_side_effect.n = 0  # type: ignore[attr-defined]
        results_side_effect.n += 1  # type: ignore[attr-defined]
        if results_side_effect.n == 1:  # type: ignore[attr-defined]

            def boom_iter():
                raise boom
                yield  # pragma: no cover

            return boom_iter()
        return iter([result])

    source._client.results.side_effect = results_side_effect

    with patch("harvest.sources.arxiv.time.sleep") as sleep:
        with patch(
            "harvest.sources.arxiv.requested_id_for",
            return_value="2106.15347",
        ):
            out = source.get_papers(["2106.15347"])

    assert out["2106.15347"] is result
    sleep.assert_called_once()
    assert sleep.call_args.args[0] == 15.0


def test_get_papers_propagates_non_429() -> None:
    source = ArxivSource()
    source._client = MagicMock()
    err = arxiv.HTTPError(url="https://export.arxiv.org/api/query", retry=1, status=500)

    def boom_iter():
        raise err
        yield  # pragma: no cover

    source._client.results.return_value = boom_iter()

    with pytest.raises(arxiv.HTTPError) as ei:
        source.get_papers(["2106.15347"])
    assert ei.value.status == 500
