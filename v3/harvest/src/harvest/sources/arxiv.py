import logging
from dataclasses import dataclass
from datetime import datetime

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
        # ArxivPaper.authors is [{"name", "affiliation"}] — arxiv.Author objects are
        # kept whole rather than flattened, so read the key, not an attribute. (This
        # line previously did `author.name` against a list[str] and raised
        # AttributeError on every call; nothing caught it because no handler calls
        # get_authors yet.) arXiv exposes no stable per-author id, hence no author_id.
        return [Author(name=a["name"]) for a in result.authors if a.get("name")]

    def get_references(self, identifier: str) -> list[Paper]:
        # arXiv has no citation graph — always empty, by design.
        return []

    def get_citations(self, identifier: str) -> list[Paper]:
        # arXiv has no citation graph — always empty, by design.
        return []

    def get_pdf_url(self, identifier: str) -> str | None:
        result = self._fetch(identifier)
        return result.pdf_url if result else None

    def _fetch(self, arxiv_id: str) -> "ArxivPaper | None":
        logger.info("fetching metadata for %s", arxiv_id)
        try:
            results = self._client.results(arxiv.Search(id_list=[arxiv_id]))
            paper = ArxivPaper.from_result(next(results))
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

    Nested library objects are kept WHOLE, not flattened to strings. They used to be
    (``authors`` as names, ``links`` as hrefs) and that quietly lost data: ``Author``
    carries ``affiliation``, and ``Link`` carries ``title``/``rel``/``content_type`` —
    the labels that identify which link is the PDF. Harvest is the expensive,
    rate-limited step, so nothing is dropped on the way in; see migration
    ``20260716200000_source_arxiv_lossless.sql``.
    """

    entry_id: str
    title: str
    authors: list[dict]
    summary: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    links: list[dict]
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
            authors=[
                {"name": a.name, "affiliation": getattr(a, "affiliation", None)}
                for a in result.authors
            ],
            summary=result.summary,
            primary_category=result.primary_category,
            categories=list(result.categories),
            pdf_url=result.pdf_url,
            links=[
                {
                    "href": link.href,
                    "title": link.title,
                    "rel": link.rel,
                    "content_type": getattr(link, "content_type", None),
                }
                for link in result.links
            ],
            published=result.published,
            updated=result.updated,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )

    def as_raw(self) -> dict:
        """The payload for ``source_arxiv.raw``.

        arxiv.Result has no ``_raw``, so the Atom XML the API actually returned is not
        recoverable from the client — this is a faithful projection of every public
        attribute instead. Datetimes are ISO strings so the dict is json-serialisable
        straight into a jsonb column.
        """
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "links": self.links,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }
