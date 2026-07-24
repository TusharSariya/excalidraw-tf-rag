"""S2 subject fetch is metadata-only; cites/refs stream via HTTP pagination."""

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


def _edge_dict(paper_id: str, *, citing: bool = True) -> dict:
    nested_key = "citingPaper" if citing else "citedPaper"
    return {
        nested_key: {
            "paperId": paper_id,
            "title": f"Paper {paper_id}",
            "year": 2020,
            "authors": [{"authorId": f"a-{paper_id}", "name": "A"}],
        }
    }


def test_related_paper_from_edge_maps_citing_paper() -> None:
    related = related_paper_from_edge(_edge_dict("cite-1", citing=True))
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


def test_iter_related_chunks_paginates_and_chunks() -> None:
    source = SemanticScholarSource()
    source._http = MagicMock()
    page1 = MagicMock()
    page1.status_code = 200
    page1.json.return_value = {
        "data": [_edge_dict("c1"), _edge_dict("c2")],
        "next": 2,
    }
    page2 = MagicMock()
    page2.status_code = 200
    page2.json.return_value = {"data": [_edge_dict("c3")]}
    source._http.get.side_effect = [page1, page2]

    chunks = list(source.iter_related_chunks("citations", "abc", chunk_size=2))
    assert [[r.paper_id for r in c] for c in chunks] == [["c1", "c2"], ["c3"]]

    first = source._http.get.call_args_list[0]
    assert first.args[0].endswith("/paper/abc/citations")
    assert first.kwargs["params"]["limit"] == RELATED_PAGE_LIMIT
    assert first.kwargs["params"]["fields"] == ",".join(RELATED_FIELDS)


def test_iter_related_chunks_references() -> None:
    source = SemanticScholarSource()
    source._http = MagicMock()
    page = MagicMock()
    page.status_code = 200
    page.json.return_value = {"data": [_edge_dict("r1", citing=False)]}
    source._http.get.return_value = page

    chunks = list(source.iter_related_chunks("references", "abc", chunk_size=1000))
    assert len(chunks) == 1
    assert chunks[0][0].paper_id == "r1"


def test_publisher_elided_references_treated_as_empty() -> None:
    source = SemanticScholarSource()
    source._http = MagicMock()
    page = MagicMock()
    page.status_code = 200
    page.json.return_value = {
        "data": None,
        "citingPaperInfo": {
            "openAccessPdf": {
                "disclaimer": "Notice: The following paper fields have been elided by the publisher: {'references'}."
            }
        },
    }
    source._http.get.return_value = page

    chunks = list(source.iter_related_chunks("references", "abc"))
    assert chunks == []


def test_transient_null_page_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harvest.sources.semantic_scholar.time.sleep", lambda _s: None)
    source = SemanticScholarSource()
    source._http = MagicMock()
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {"data": None}
    good = MagicMock()
    good.status_code = 200
    good.json.return_value = {"data": [_edge_dict("r1", citing=False)]}
    source._http.get.side_effect = [bad, good]

    chunks = list(source.iter_related_chunks("references", "abc"))
    assert [r.paper_id for r in chunks[0]] == ["r1"]
    assert source._http.get.call_count == 2


def test_get_papers_treats_no_valid_ids_as_not_found() -> None:
    source = SemanticScholarSource()
    source._client = MagicMock()
    source._client.get_papers.side_effect = BadQueryParametersException("No valid paper ids given")

    out = source.get_papers(["DOI:10.1145/missing"])
    assert out == {"DOI:10.1145/missing": None}


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
    related = related_paper_from_edge(
        {
            "citingPaper": {
                "paperId": "x",
                "title": "T",
                "citations": [],
            }
        }
    )
    assert isinstance(related, RelatedPaper)
    assert related.paper_id == "x"
