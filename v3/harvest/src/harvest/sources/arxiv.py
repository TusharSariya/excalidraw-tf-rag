import logging
import re
import time

import arxiv

from .base import PaperSource

logger = logging.getLogger("harvest.arxiv")

# Strip a trailing version suffix (v1, v2, …) for matching request ids that omit it.
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)

# Client-level pacing (arxiv API ToU: ~1 request / 3s). num_retries covers brief
# blips; outer 429 loop below covers sustained rate limits.
_CLIENT_DELAY_SECONDS = 3.0
_CLIENT_NUM_RETRIES = 5

# After the Client exhausts its own retries on HTTP 429, back off and try again
# before propagating — same philosophy as OpenAlex's retry_http_codes.
_OUTER_429_ATTEMPTS = 4
_OUTER_429_BACKOFF_SECONDS = (15.0, 30.0, 60.0)


def _bare_id(arxiv_id: str) -> str:
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def requested_id_for(result: arxiv.Result, requested: set[str]) -> str | None:
    """Map a search Result back to one of the ids we asked for."""
    short = result.get_short_id()
    bare = _bare_id(short)
    candidates = {short, bare}
    for req in requested:
        if req in candidates or _bare_id(req) == bare:
            return req
    return None


class ArxivSource(PaperSource):
    """PaperSource implementation backed by the arxiv.org API."""

    def __init__(self) -> None:
        self._client = arxiv.Client(
            delay_seconds=_CLIENT_DELAY_SECONDS,
            num_retries=_CLIENT_NUM_RETRIES,
        )

    def get_papers(self, identifiers: list[str]) -> dict[str, arxiv.Result | None]:
        """Batch-fetch by arXiv id; missing ids map to None (terminal, don't retry).

        Transient HTTP 429s are retried with a long outer backoff after the
        Client's own retries are exhausted. Other errors (and 429s that still
        fail) propagate so the worker leaves messages unarchived for redelivery.
        """
        if not identifiers:
            return {}

        logger.info("fetching papers for %s", identifiers)
        for attempt in range(_OUTER_429_ATTEMPTS):
            try:
                return self._fetch_once(identifiers)
            except arxiv.HTTPError as err:
                if err.status != 429 or attempt >= _OUTER_429_ATTEMPTS - 1:
                    raise
                delay = _OUTER_429_BACKOFF_SECONDS[
                    min(attempt, len(_OUTER_429_BACKOFF_SECONDS) - 1)
                ]
                logger.warning(
                    "arxiv HTTP 429 on attempt %d/%d — sleeping %.0fs before retry",
                    attempt + 1,
                    _OUTER_429_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
        return {i: None for i in identifiers}  # pragma: no cover — loop always returns/raises

    def _fetch_once(self, identifiers: list[str]) -> dict[str, arxiv.Result | None]:
        out: dict[str, arxiv.Result | None] = {i: None for i in identifiers}
        remaining = set(identifiers)
        results = self._client.results(arxiv.Search(id_list=list(identifiers)))
        for result in results:
            key = requested_id_for(result, remaining)
            if key is None:
                logger.warning(
                    "arxiv result %s matched no outstanding request id",
                    result.get_short_id(),
                )
                continue
            out[key] = result
            remaining.discard(key)
        return out
