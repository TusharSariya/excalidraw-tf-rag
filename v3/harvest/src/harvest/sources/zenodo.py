import logging

import requests

from .base import PaperSource
from .models.zenodo import ZenodoRecord

logger = logging.getLogger("harvest.zenodo")

_BASE_URL = "https://zenodo.org/api/records"

# Per-request timeout (seconds). A single search is a small GET; a slow response
# is treated as transient and raised so pgmq redelivers the batch.
_TIMEOUT_SECONDS = 30.0


def _normalize_doi(identifier: str) -> str:
    """A seed carries the DOI bare (``10.x/…``); the search ``q`` wants it bare too.

    Tolerate a DOI handed in as a ``https://doi.org/…`` or ``doi:…`` URL by
    stripping the prefix. Case is left untouched — Zenodo DOIs are matched by the
    search field verbatim, and lowercasing a non-Zenodo DOI would gain nothing.
    """
    key = identifier.strip()
    low = key.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key


class ZenodoSource(PaperSource):
    """PaperSource implementation backed by the Zenodo legacy records search API.

    ``identifier`` is a DOI (bare ``10.x/…``). Zenodo is reached by DOI only —
    there is no arXiv-id, title, or OpenAlex-id lookup — so seeds that carry no
    DOI never reach this source (the ``{"any_of": ["doi"]}`` dispatch filter).

    **No batch endpoint.** The search endpoint takes one ``q`` per request; there
    is no first-class multi-id / filter lookup, so ``get_papers`` iterates serially
    while keeping the ``list -> dict`` contract.

    **Concept vs version DOI.** A seed DOI may be a *concept* (all-versions) DOI,
    which lives in a record's ``conceptdoi`` field rather than ``doi`` (the
    version DOI). A naive ``q=doi:"{doi}"`` returns ``total=0`` for a concept DOI,
    so the query ORs both fields: ``doi:"{doi}" OR conceptdoi:"{doi}"``.

    **Many paper DOIs are not Zenodo-hosted** — those return ``hits.total == 0``,
    which is a valid *not-found* (``None``), NOT an error. Transient HTTP (429 rate
    limit, 5xx) raises so the worker leaves the message unarchived and pgmq
    redelivers after the visibility timeout — no in-process backoff, which would
    risk overrunning ``worker.VT_SECONDS``.
    """

    def __init__(self) -> None:
        self._session = requests.Session()

    def get_papers(self, identifiers: list[str]) -> dict[str, ZenodoRecord | None]:
        """Serial-fetch by DOI; not-found DOIs map to None (terminal, don't retry).

        No native batch endpoint exists, so this loops one DOI at a time. A search
        with ``hits.total == 0`` (paper not Zenodo-hosted) records None; transient
        429/5xx raise so the worker leaves the messages unarchived for pgmq
        redelivery. Retrying in-process instead would risk blocking past
        ``worker.VT_SECONDS``, which lets pgmq hand the same batch to a second
        worker while this one is still sleeping.
        """
        out: dict[str, ZenodoRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)
        for req in identifiers:
            doi = _normalize_doi(req)
            if not doi:
                logger.warning("blank DOI for request %r; recording not_found", req)
                continue

            # OR both fields so a concept (all-versions) DOI, which lives in
            # `conceptdoi` not `doi`, still resolves. Values are quoted for exact match.
            resp = self._session.get(
                _BASE_URL,
                params={"q": f'doi:"{doi}" OR conceptdoi:"{doi}"'},
                timeout=_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                # 429 (rate limit) and 5xx are transient — raise so pgmq redelivers.
                logger.error("fetch failed (HTTP %s) for %s", resp.status_code, req)
                resp.raise_for_status()
                # Non-200, non-error status (e.g. an unexpected 3xx) — treat as
                # transient rather than silently record a miss.
                raise RuntimeError(f"unexpected HTTP {resp.status_code} for {req}")

            body = resp.json()
            hits = body.get("hits") or {}
            if not hits.get("total"):
                # total == 0 (or absent): paper not Zenodo-hosted — valid not_found.
                logger.warning("no Zenodo record for %s (not found)", req)
                continue

            entries = hits.get("hits") or []
            if not entries:
                continue
            out[req] = ZenodoRecord.from_result(entries[0])

        return out
