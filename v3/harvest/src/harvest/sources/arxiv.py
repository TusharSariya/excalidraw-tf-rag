import logging
import re

import arxiv

from .base import PaperSource

logger = logging.getLogger("harvest.arxiv")

# Strip a trailing version suffix (v1, v2, …) for matching request ids that omit it.
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)

# Client-level pacing (arxiv API ToU: ~1 request / 3s). num_retries covers brief
# blips; sustained rate limits are left to pgmq redelivery, which already spaces
# attempts by worker.VT_SECONDS (120s) — a longer backoff than we could sleep
# here without overrunning the visibility timeout and losing the batch.
_CLIENT_DELAY_SECONDS = 3.0
_CLIENT_NUM_RETRIES = 5


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

        Errors propagate — including HTTP 429s that outlive the Client's own
        retries — so the worker leaves the messages unarchived and pgmq
        redelivers them. Retrying in-process instead would risk blocking past
        worker.VT_SECONDS, which lets pgmq hand the same batch to a second
        worker while this one is still sleeping.
        """
        if not identifiers:
            return {}

        logger.info("fetching papers for %s", identifiers)
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
