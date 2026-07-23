"""Unit tests for Semantic Scholar request-id remapping (no network)."""

from __future__ import annotations

from harvest.sources.semantic_scholar import requested_id_for
from semanticscholar.Paper import Paper


def test_requested_id_for_matches_arxiv_namespace() -> None:
    paper = Paper(
        {
            "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "externalIds": {"ArXiv": "1706.03762", "CorpusId": 13756489},
        }
    )
    requested = {"ARXIV:1706.03762", "DOI:10.1145/missing"}
    assert requested_id_for(paper, requested) == "ARXIV:1706.03762"


def test_requested_id_for_matches_doi_namespace() -> None:
    paper = Paper(
        {
            "paperId": "abc",
            "externalIds": {"DOI": "10.1145/3292500.3330919"},
        }
    )
    assert requested_id_for(paper, {"DOI:10.1145/3292500.3330919"}) == (
        "DOI:10.1145/3292500.3330919"
    )


def test_requested_id_for_matches_bare_paper_id() -> None:
    paper = Paper({"paperId": "deadbeef", "externalIds": {}})
    assert requested_id_for(paper, {"deadbeef", "ARXIV:1"}) == "deadbeef"


def test_requested_id_for_returns_none_when_unmatched() -> None:
    paper = Paper(
        {
            "paperId": "abc",
            "externalIds": {"ArXiv": "1706.03762"},
        }
    )
    assert requested_id_for(paper, {"ARXIV:9999.99999"}) is None
