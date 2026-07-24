"""S2 subject fetch is metadata-only; cites/refs stream via paginated endpoints."""

from unittest.mock import MagicMock

import pytest
from harvest.sources.models.s2 import RelatedPaper
from harvest.sources.semantic_scholar import (
    RELATED_FIELDS,
    RELATED_PAGE_LIMIT,
    SUBJECT_FIELDS,
    SemanticScholarSource,
    related_paper_from_edge,
)
from semanticscholar.SemanticScholarException import BadQueryParametersException


def _subject_paper(
    *,
    paper_id: str = "abc",
    external_ids: dict | None = None,
):
    p = MagicMock()
    p.paperId = paper_id
    p.externalIds = external_ids or {"DOI": "10.1145/example"}
    p.raw_data = {
        "paperId": paper_id,
        "externalIds": p.externalIds,
        "title": "Example",
        "citationCount": 2,
        "referenceCount": 1,
    }
    return p


def _edge(paper_id: str, *, citing: bool = True):
    nested_key = "citingPaper" if citing else "citedPaper"
    item = MagicMock()
    item.paper = MagicMock()
    item.paper.raw_data = {
        "paperId": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2020,
        "authors": [{"authorId": f"a-{paper_id}", "name": "A"}],
    }
    item.raw_data = {nested_key: item.paper.raw_data}
    return item


def test_related_paper_from_edge_maps_citing_paper() -> None:
    related = related_paper_from_edge(_edge("cite-1", citing=True))
    assert related is not None
    assert related.paper_id == "cite-1"
    assert related.title == "Paper cite-1"
    assert related.authors is not None
    assert related.authors[0].author_id == "a-cite-1"


def test_get_papers_uses_subject_fields_only() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    paper = _subject_paper()
    source._client.get_papers.return_value = ([paper], [])

    out = source.get_papers(["DOI:10.1145/example"])
    rec = out["DOI:10.1145/example"]
    assert rec is not None
    assert rec.paper_id == "abc"
    assert rec.citations is None
    assert rec.references is None
    assert rec.citation_count == 2
    assert rec.reference_count == 1

    kwargs = source._client.get_papers.call_args.kwargs
    assert kwargs["fields"] == SUBJECT_FIELDS
    assert not any(f.startswith("citations.") for f in kwargs["fields"])
    assert not any(f.startswith("references.") for f in kwargs["fields"])
    source._client.get_paper_citations.assert_not_called()
    source._client.get_paper_references.assert_not_called()


def test_iter_related_chunks_paginates_and_chunks() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_paper_citations.return_value = [
        _edge("c1"),
        _edge("c2"),
        _edge("c3"),
    ]

    chunks = list(source.iter_related_chunks("citations", "abc", chunk_size=2))
    assert [[r.paper_id for r in c] for c in chunks] == [["c1", "c2"], ["c3"]]

    cite_kw = source._client.get_paper_citations.call_args.kwargs
    assert cite_kw["fields"] == RELATED_FIELDS
    assert cite_kw["limit"] == RELATED_PAGE_LIMIT
    assert source._client.get_paper_citations.call_args.args[0] == "abc"


def test_iter_related_chunks_references() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_paper_references.return_value = [_edge("r1", citing=False)]

    chunks = list(source.iter_related_chunks("references", "abc", chunk_size=1000))
    assert len(chunks) == 1
    assert chunks[0][0].paper_id == "r1"


def test_open_related_retries_null_page_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harvest.sources.semantic_scholar.time.sleep", lambda _s: None)
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_paper_references.side_effect = [
        TypeError("'NoneType' object is not iterable"),
        [_edge("r1", citing=False)],
    ]

    chunks = list(source.iter_related_chunks("references", "abc"))
    assert [r.paper_id for r in chunks[0]] == ["r1"]
    assert source._client.get_paper_references.call_count == 2


def test_mid_stream_null_page_raises_for_redelivery() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()

    def _broken():
        yield _edge("c1")
        raise TypeError("'NoneType' object is not iterable")

    source._client.get_paper_citations.return_value = _broken()

    with pytest.raises(TypeError, match="NoneType"):
        list(source.iter_related_chunks("citations", "abc"))


def test_get_papers_treats_no_valid_ids_as_not_found() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = BadQueryParametersException("No valid paper ids given")

    out = source.get_papers(["DOI:10.1145/missing"])
    assert out == {"DOI:10.1145/missing": None}
    source._client.get_paper_citations.assert_not_called()


def test_get_papers_rethrows_non_id_query_error() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = BadQueryParametersException(
        "Unrecognized or unsupported fields: [bogus]"
    )

    with pytest.raises(BadQueryParametersException, match="Unrecognized"):
        source.get_papers(["DOI:10.1145/a"])


def test_get_papers_fetches_serially() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = [
        ([_subject_paper(external_ids={"DOI": "10.1/a"})], []),
        ([_subject_paper(paper_id="b", external_ids={"DOI": "10.1/b"})], []),
    ]

    source.get_papers(["DOI:10.1/a", "DOI:10.1/b"])
    assert source._client.get_papers.call_count == 2
    assert source._client.get_papers.call_args_list[0].kwargs["paper_ids"] == ["DOI:10.1/a"]
    assert source._client.get_papers.call_args_list[1].kwargs["paper_ids"] == ["DOI:10.1/b"]


def test_related_paper_from_edge_filters_unknown_keys() -> None:
    item = MagicMock()
    item.paper = MagicMock()
    item.paper.raw_data = {
        "paperId": "x",
        "title": "T",
        "citations": [],  # not a RelatedPaper key — filtered
    }
    item.raw_data = {"citingPaper": item.paper.raw_data}
    related = related_paper_from_edge(item)
    assert isinstance(related, RelatedPaper)
    assert related.paper_id == "x"
