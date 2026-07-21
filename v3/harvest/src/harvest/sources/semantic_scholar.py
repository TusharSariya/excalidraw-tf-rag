import logging

from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import ObjectNotFoundException

from .base import PaperSource
from .models.s2 import S2Record

logger = logging.getLogger("harvest.semantic_scholar")


class SemanticScholarSource(PaperSource):
    """PaperSource implementation backed by the Semantic Scholar Graph API.

    ``identifier`` is a Semantic Scholar external paper id, already namespaced
    by the caller (e.g. "ARXIV:1706.03762" or "DOI:10.1145/..."), and passed to
    the Graph API verbatim.
    """

    def __init__(self) -> None:
        self._client = SemanticScholar()

    def get_paper(self, identifier: str) -> S2Record | None:
        """Fetch one paper, or None if S2 genuinely has no record for it.

        Only a not-found is swallowed to None (a terminal, don't-retry outcome).
        Any other ``SemanticScholarException`` (rate limit, network, 5xx) is
        transient, so it propagates — the worker leaves the message unarchived
        and pgmq redelivers it, rather than mislabelling the seed not-found.
        """
        logger.info("fetching paper for %s", identifier)
        try:
            paper = self._client.get_paper(identifier)
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", identifier)
            return None

        return S2Record.from_sdk(paper)
