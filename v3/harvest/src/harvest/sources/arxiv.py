import logging

import arxiv

logger = logging.getLogger("harvest.arxiv")


class ArxivSource:
    """PaperSource implementation backed by the arxiv.org API."""

    def __init__(self) -> None:
        self._client = arxiv.Client()

    def get_paper(self, identifier: str) -> arxiv.Result | None:
        try:
            return next(self._client.results(arxiv.Search(id_list=[identifier])))
        except StopIteration:
            logger.warning("no result for %s (not found)", identifier)
            return None
