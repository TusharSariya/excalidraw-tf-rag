import logging

import internetarchive
from requests.exceptions import HTTPError

from .base import PaperSource
from .models.internet_archive import InternetArchiveItem

logger = logging.getLogger("harvest.internet_archive")


class InternetArchiveSource(PaperSource):
    """PaperSource implementation backed by the Internet Archive metadata API.

    **`identifier` is an IA item id (the seed's ``ia_identifier``) — nothing else.**
    ``internetarchive.get_item(identifier)`` fetches ``GET https://archive.org/metadata/{identifier}``
    anonymously (no API key is needed, and none is configured for read/metadata), and
    exposes the parsed response on ``Item.item_metadata``.

    There is **no native batch / multi-id endpoint** — metadata is one identifier per
    request — so ``get_papers`` is a serial loop that keeps the ``list -> dict`` contract.

    Identity is never resolved by title search: IA holds many OCR scans with colliding
    titles (and this corpus's "Attention Is All You Need" is one such low-trust scan), so
    a fuzzy match would upsert the wrong artefact as confidently as the right one. Give it
    the exact ``ia_identifier`` or it returns None.

    Missing item -> None (terminal; the worker marks not_found). A nonexistent id returns
    HTTP 200 with an empty object (``Item.exists is False``), which is the not_found
    signal. Transient HTTP (429 / 5xx) and other request errors propagate so the worker
    leaves the message unarchived and pgmq redelivers after the visibility timeout.

    This is metadata only: the PDF url is *constructed* from the item id and the chosen
    PDF filename (see ``InternetArchiveItem.pdf_url``); the binary is never fetched.
    """

    def get_papers(self, identifiers: list[str]) -> dict[str, InternetArchiveItem | None]:
        """Serial fetch by IA identifier; missing ids map to None (terminal, don't retry).

        No batch endpoint exists, so each id is a separate ``get_item`` call. Every
        requested id is a key in the returned dict. Transient HTTP errors raise.
        """
        out: dict[str, InternetArchiveItem | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)
        for req in identifiers:
            try:
                item = internetarchive.get_item(req)
            except HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning("no result for %s (not found)", req)
                    continue
                logger.error("fetch failed (HTTP %s): %s", status, e)
                raise

            if not item.exists:
                logger.warning("no result for %s (not found)", req)
                continue

            out[req] = InternetArchiveItem.from_result(dict(item.item_metadata))

        return out
