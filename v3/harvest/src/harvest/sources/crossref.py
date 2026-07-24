import logging
import os

import requests

from .base import PaperSource
from .models.crossref import CrossrefWork

logger = logging.getLogger("harvest.crossref")

# Crossref's native multi-id endpoint: /works with a DOI OR-filter returns
# message.items[] for every match in ONE request. This is the batch path the
# harvester uses — never a per-DOI loop.
CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Per-request DOI cap. The filter is an OR over DOIs (filter=doi:A,doi:B,...);
# 20 keeps the request URL well inside limits and matches the probed batch size.
DOI_BATCH = 20

# A single request's timeout. Sustained rate limits / 5xx are left to pgmq
# redelivery (worker.VT_SECONDS), not an in-process sleep that could overrun the
# visibility timeout and lose the batch.
REQUEST_TIMEOUT = 30


def requested_id_for(item: dict, requested: set[str]) -> str | None:
    """Map a ``message.items[]`` object back to one of the DOIs we asked for.

    Crossref echoes the DOI LOWERCASED even when the query DOI was mixed-case, so
    match case-folded.
    """
    doi = (item.get("DOI") or "").lower()
    if not doi:
        return None
    for req in requested:
        if req.lower() == doi:
            return req
    return None


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class CrossrefSource(PaperSource):
    """PaperSource implementation backed by the Crossref REST API.

    ``identifier`` is a DOI (bare ``10.x`` form, as carried on ``seeds.doi``).
    Crossref's ``/works?filter=doi:A,doi:B,...`` resolves a batch of DOIs in one
    request; DOIs with no matching record simply do not appear in
    ``message.items[]`` and map to None (terminal — the worker marks not_found).

    ``mailto`` is Crossref's optional "polite pool" hint, NOT an auth key: an
    unset value does not make the source unreachable, so it is included only when
    present (from the constructor or ``CROSSREF_MAILTO``). No API key is required.
    """

    def __init__(self, mailto: str | None = None) -> None:
        self._mailto = mailto or os.getenv("CROSSREF_MAILTO") or None
        if not self._mailto:
            logger.warning(
                "no CROSSREF_MAILTO: running outside Crossref's polite pool. Set "
                "CROSSREF_MAILTO for more predictable rate limits."
            )
        self._session = requests.Session()
        ua = "harvest/crossref"
        if self._mailto:
            ua = f"{ua} (mailto:{self._mailto})"
        self._session.headers.update({"User-Agent": ua})

    def get_papers(self, identifiers: list[str]) -> dict[str, CrossrefWork | None]:
        """Batch-fetch by DOI; missing DOIs map to None (terminal, don't retry).

        Chunked to ``DOI_BATCH`` and issued through the ``/works`` DOI-filter
        endpoint. Transient HTTP (429/5xx) raises so the worker leaves the
        messages unarchived and pgmq redelivers them after the visibility
        timeout; a 404 on the route is treated as "nothing matched" for that
        chunk.
        """
        out: dict[str, CrossrefWork | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)

        for chunk in _chunks(list(identifiers), DOI_BATCH):
            remaining = set(chunk)
            params: dict[str, str | int] = {
                "filter": ",".join(f"doi:{doi}" for doi in chunk),
                "rows": len(chunk),
            }
            if self._mailto:
                params["mailto"] = self._mailto

            resp = self._session.get(CROSSREF_WORKS_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                logger.warning("no result for crossref batch %s (not found)", chunk)
                continue
            resp.raise_for_status()

            items = (resp.json().get("message") or {}).get("items") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = requested_id_for(item, remaining)
                if key is None:
                    logger.warning(
                        "crossref item %s matched no outstanding request id",
                        item.get("DOI"),
                    )
                    continue
                out[key] = CrossrefWork.from_result(item)
                remaining.discard(key)

        return out
