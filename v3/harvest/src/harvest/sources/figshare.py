import logging

import requests
from requests.exceptions import HTTPError

from .base import PaperSource
from .models.figshare import FigshareArticle

logger = logging.getLogger("harvest.figshare")

_API_BASE = "https://api.figshare.com/v2"
_SEARCH_URL = f"{_API_BASE}/articles/search"

# Public reads need no auth. A short per-request timeout keeps a stalled socket
# from pinning a worker past the pgmq visibility timeout (worker.VT_SECONDS).
_TIMEOUT_SECONDS = 30
_USER_AGENT = "excalidraw-tf-rag-harvest/figshare (+https://github.com/TusharSariyaOrg)"

# Transient statuses left to pgmq redelivery rather than an in-process backoff:
# raising surfaces them so the worker leaves the message unarchived (redelivered
# after VT_SECONDS), which is a longer, safer wait than we could sleep here.
_TRANSIENT = {429, 500, 502, 503, 504}


class FigshareSource(PaperSource):
    """PaperSource implementation backed by the public figshare v2 API.

    ``identifier`` is a **publication DOI** (the seed's ``doi``). figshare has no
    native multi-id/filter endpoint, so ``get_papers`` runs a serial loop while
    keeping the ``list -> dict`` contract.

    Resolution is two hops, forced by the API shape:

    1. ``POST /articles/search`` with ``{"resource_doi": <doi>}``. figshare mints
       its OWN DOI per deposit (``…g001``), so the seed's publication DOI lives in
       ``resource_doi`` — searching ``{"doi": <doi>}`` returns nothing. An empty
       result is a terminal not-found (``None``); the seed simply has no figshare
       deposit, which is expected for most publishers.
    2. ``GET /articles/{id}`` for the first hit, because search list items are a
       SUBSET of the full record (no files / authors / description / license) and
       harvest needs the deposit file url + rich metadata.

    Identity is exact, never fuzzy: we search by the seed's DOI on ``resource_doi``
    and key the result on that same requested DOI — no title search (see the
    ``OpenAlexSource`` docstring for why that is a trap, not a fallback).
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def get_papers(self, identifiers: list[str]) -> dict[str, FigshareArticle | None]:
        """Serial-fetch by publication DOI; missing ids map to None (terminal).

        No native batch endpoint exists, so this loops one DOI at a time. Only a
        genuine not-found (empty search or 404) is recorded as None. Transient HTTP
        (429/5xx) or a network error propagates so the worker leaves the message
        unarchived for pgmq redelivery.
        """
        out: dict[str, FigshareArticle | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching papers for %s", identifiers)
        for doi in identifiers:
            article = self._fetch_one(doi)
            if article is not None:
                out[doi] = article
        return out

    def _fetch_one(self, doi: str) -> FigshareArticle | None:
        """Resolve one publication DOI to its (first) figshare deposit, or None."""
        article_id = self._search_article_id(doi)
        if article_id is None:
            logger.warning("no figshare deposit for %s (not found)", doi)
            return None
        return self._get_article(article_id, doi)

    def _search_article_id(self, doi: str) -> int | None:
        """POST /articles/search on ``resource_doi``; return the first hit's id."""
        resp = self._session.post(_SEARCH_URL, json={"resource_doi": doi}, timeout=_TIMEOUT_SECONDS)
        self._raise_for_status(resp, f"search resource_doi={doi}")
        results = resp.json()
        if not isinstance(results, list) or not results:
            return None
        return results[0].get("id")

    def _get_article(self, article_id: int, doi: str) -> FigshareArticle | None:
        """GET /articles/{id} for the full record. 404 -> None (terminal)."""
        resp = self._session.get(f"{_API_BASE}/articles/{article_id}", timeout=_TIMEOUT_SECONDS)
        if resp.status_code == 404:
            logger.warning("figshare article %s (from %s) 404 on full fetch", article_id, doi)
            return None
        self._raise_for_status(resp, f"article id={article_id}")
        return FigshareArticle.from_result(resp.json())

    @staticmethod
    def _raise_for_status(resp: requests.Response, what: str) -> None:
        """Raise on transient/other error statuses; leave 2xx alone.

        A transient 429/5xx is logged and re-raised so pgmq redelivers; any other
        non-2xx also raises (via ``raise_for_status``) rather than being silently
        swallowed. 404 is handled by callers before this is reached.
        """
        if resp.status_code in _TRANSIENT:
            logger.error("figshare %s failed (HTTP %s)", what, resp.status_code)
        try:
            resp.raise_for_status()
        except HTTPError:
            raise
