import logging
from dataclasses import dataclass
from datetime import datetime
import pprint

import arxiv

from .base import Author, Paper

logger = logging.getLogger("harvest.arxiv")


class ArxivSource:
    """PaperSource implementation backed by the arxiv.org API."""

    def __init__(self) -> None:
        self._client = arxiv.Client()

    def get_paper(self, identifier: str) -> Paper | None:
        result = self._fetch(identifier)
        if result is None:
            return None
        return Paper(
            title=result.title,
            arxiv_id=identifier,
            pdf_url=result.pdf_url,
        )

    def get_authors(self, identifier: str) -> list[Author]:
        result = self._fetch(identifier)
        if result is None:
            return []
        return [Author(name=author.name) for author in result.authors]

    def get_references(self, identifier: str) -> list[Paper]:
        # arXiv has no citation graph — always empty, by design.
        return []

    def get_citations(self, identifier: str) -> list[Paper]:
        # arXiv has no citation graph — always empty, by design.
        return []

    def get_pdf_url(self, identifier: str) -> str | None:
        result = self._fetch(identifier)
        return result.pdf_url if result else None

    def _fetch(self, arxiv_id: str) -> arxiv.Result | None:
        logger.info("fetching metadata for %s", arxiv_id)
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            logger.info("search: %s", search)
            results = self._client.results(search)
            logger.info("results: %s", results)
            paper =next(results)
            logger.info("paper: %s", paper)
            arxiv_paper = ArxivPaper.from_result(paper)
            logger.info("arxiv_paper: %s", pprint.pformat(arxiv_paper, sort_dicts=False, width=100))
            return arxiv_paper

        except StopIteration:
            logger.warning("no result for arxiv_id=%s (not found)", arxiv_id)
            return None
        except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError) as e:
            logger.error("fetch failed after retries: %s", e)
            return None
        logger.info("metadata fetched: title=%r pdf_url=%s", paper.title, paper.pdf_url)
        return paper


@dataclass
class ArxivPaper:
    """Every field from an ``arxiv.Result``, kept before projecting to ``Paper``.

    Nested library objects are flattened: ``authors`` becomes a list of names
    and ``links`` a list of hrefs.
    """

    entry_id: str
    title: str
    authors: list[str]
    summary: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    links: list[str]
    published: datetime
    updated: datetime
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    @classmethod
    def from_result(cls, result: arxiv.Result) -> "ArxivPaper":
        return cls(
            entry_id=result.entry_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            summary=result.summary,
            primary_category=result.primary_category,
            categories=list(result.categories),
            pdf_url=result.pdf_url,
            links=[link.href for link in result.links],
            published=result.published,
            updated=result.updated,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )
