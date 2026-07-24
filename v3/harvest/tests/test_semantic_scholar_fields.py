"""S2 fetch uses a rich→thin→meta field ladder at n=1 (not full Paper.FIELDS)."""

from unittest.mock import MagicMock

import pytest
from harvest.sources.semantic_scholar import (
    META_FIELDS,
    RICH_FIELDS,
    THIN_FIELDS,
    SemanticScholarSource,
)
from semanticscholar.SemanticScholarException import BadQueryParametersException


def _paper(
    *,
    paper_id: str = "abc",
    external_ids: dict | None = None,
    citations: list | None = None,
):
    p = MagicMock()
    p.paperId = paper_id
    p.externalIds = external_ids or {"DOI": "10.1145/example"}
    p.citations = citations
    p.references = []
    p.embedding = None
    p.tldr = None
    p.title = "Example"
    return p


def test_get_papers_uses_rich_fields_first() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    paper = _paper()
    source._client.get_papers.return_value = ([paper], [])

    # from_sdk is heavy; stub it
    import harvest.sources.semantic_scholar as mod

    mod.S2Record.from_sdk = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

    source.get_papers(["DOI:10.1145/example"])

    kwargs = source._client.get_papers.call_args.kwargs
    assert kwargs["paper_ids"] == ["DOI:10.1145/example"]
    assert kwargs["fields"] == RICH_FIELDS
    assert "citations.abstract" in kwargs["fields"]
    assert "embedding" in kwargs["fields"]
    assert "tldr" in kwargs["fields"]


def test_get_papers_falls_back_thin_then_meta_on_size_error() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    size_err = BadQueryParametersException(
        "Response would exceed maximum size. Please retry with 1 or fewer ids"
    )
    paper = _paper()
    source._client.get_papers.side_effect = [
        size_err,  # rich
        size_err,  # thin
        ([paper], []),  # meta
    ]

    import harvest.sources.semantic_scholar as mod

    mod.S2Record.from_sdk = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

    out = source.get_papers(["DOI:10.1145/example"])
    assert out["DOI:10.1145/example"] is not None

    field_sets = [c.kwargs["fields"] for c in source._client.get_papers.call_args_list]
    assert field_sets == [RICH_FIELDS, THIN_FIELDS, META_FIELDS]


def test_get_papers_treats_no_valid_ids_as_not_found() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = BadQueryParametersException("No valid paper ids given")

    out = source.get_papers(["DOI:10.1145/missing"])
    assert out == {"DOI:10.1145/missing": None}


def test_get_papers_rethrows_non_size_query_error() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = BadQueryParametersException(
        "Unrecognized or unsupported fields: [citations.authors.name]"
    )

    with pytest.raises(BadQueryParametersException, match="Unrecognized"):
        source.get_papers(["DOI:10.1145/a"])


def test_get_papers_fetches_serially() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = [
        ([_paper(external_ids={"DOI": "10.1/a"})], []),
        ([_paper(paper_id="b", external_ids={"DOI": "10.1/b"})], []),
    ]

    import harvest.sources.semantic_scholar as mod

    mod.S2Record.from_sdk = MagicMock(side_effect=[MagicMock(), MagicMock()])  # type: ignore[method-assign]

    source.get_papers(["DOI:10.1/a", "DOI:10.1/b"])
    assert source._client.get_papers.call_count == 2
    assert source._client.get_papers.call_args_list[0].kwargs["paper_ids"] == ["DOI:10.1/a"]
    assert source._client.get_papers.call_args_list[1].kwargs["paper_ids"] == ["DOI:10.1/b"]
