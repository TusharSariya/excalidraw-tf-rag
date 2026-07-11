import logging

from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import (
    ObjectNotFoundException,
    SemanticScholarException,
)

from .base import Author, Paper

logger = logging.getLogger("harvest.semantic_scholar")


class SemanticScholarSource:
    """PaperSource implementation backed by the Semantic Scholar Graph API.

    `identifier` is treated as an arXiv ID (e.g. "1706.03762") and mapped
    to Semantic Scholar's "ARXIV:<id>" paper ID format.
    """

    def __init__(self) -> None:
        self._client = SemanticScholar()

    def get_paper(self, identifier: str) -> Paper | None:
        schpaper_id = self._schpaper_id(identifier)
        logger.info("fetching paper for %s", schpaper_id)
        try:
            paper = self._client.get_paper(schpaper_id)
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", schpaper_id)
            return None
        except SemanticScholarException as e:
            logger.error("fetch failed: %s", e)
            return None

        external_ids = paper.externalIds or {}
        open_access_pdf = paper.openAccessPdf or {}
        return Paper(
            title=paper.title,
            arxiv_id=external_ids.get("ArXiv"),
            doi=external_ids.get("DOI"),
            pdf_url=open_access_pdf.get("url") or None,
        )

    def get_authors(self, identifier: str) -> list[Author]:
        schpaper_id = self._schpaper_id(identifier)
        logger.info("fetching authors for %s", schpaper_id)
        try:
            authors = self._client.get_paper_authors(schpaper_id, fields=["name", "authorId"])
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", schpaper_id)
            return []
        except SemanticScholarException as e:
            logger.error("fetch failed: %s", e)
            return []
        return [Author(name=a.name, author_id=a.authorId) for a in authors]

    def get_references(self, identifier: str) -> list[Paper]:
        schpaper_id = self._schpaper_id(identifier)
        logger.info("fetching references for %s", schpaper_id)
        try:
            references = self._client.get_paper_references(schpaper_id, fields=["title", "paperId"])
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", schpaper_id)
            return []
        except SemanticScholarException as e:
            logger.error("fetch failed: %s", e)
            return []
        return [Paper(title=r.paper.title) for r in references]

    def get_citations(self, identifier: str) -> list[Paper]:
        schpaper_id = self._schpaper_id(identifier)
        logger.info("fetching citations for %s", schpaper_id)
        try:
            citations = self._client.get_paper_citations(
                schpaper_id, fields=["title", "paperId"], limit=100
            )
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", schpaper_id)
            return []
        except SemanticScholarException as e:
            logger.error("fetch failed: %s", e)
            return []
        return [Paper(title=c.paper.title) for c in citations.items]

    def get_pdf_url(self, identifier: str) -> str | None:
        paper = self.get_paper(identifier)
        return paper.pdf_url if paper else None

    @staticmethod
    def _schpaper_id(identifier: str) -> str:
        return f"ARXIV:{identifier}"
