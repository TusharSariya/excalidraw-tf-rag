"""Batch get_papers must request Paper.FIELDS, not the SDK SEARCH_FIELDS default."""

from unittest.mock import MagicMock

from harvest.sources.semantic_scholar import SemanticScholarSource
from semanticscholar.Paper import Paper


def test_get_papers_requests_full_fields() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.return_value = ([], [])

    source.get_papers(["DOI:10.1145/example"])

    kwargs = source._client.get_papers.call_args.kwargs
    assert kwargs["fields"] == Paper.FIELDS
    assert "citations" in kwargs["fields"]
    assert "references" in kwargs["fields"]
    assert "embedding" in kwargs["fields"]
