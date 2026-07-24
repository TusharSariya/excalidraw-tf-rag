import logging
import os

import requests

from .base import PaperSource
from .models.unpaywall import UnpaywallRecord

logger = logging.getLogger("harvest.unpaywall")

_BASE_URL = "https://api.unpaywall.org/v2/"

# Unpaywall requires a valid email on EVERY request (query param). It is a hard
# credential, not an optional polite-pool key: without it every call 422s.
_EMAIL_ENV = "UNPAYWALL_EMAIL"

# Per-request timeout (seconds). A single DOI lookup is a small GET; a slow
# response is treated as transient and raised so pgmq redelivers the batch.
_TIMEOUT_SECONDS = 30.0


def _normalize_doi(identifier: str) -> str:
    """A seed carries the DOI bare (``10.x/…``); Unpaywall's ``/v2/{doi}`` wants it bare too.

    Tolerate a DOI handed in as a ``https://doi.org/…`` or ``doi:…`` URL by
    stripping the prefix, and lowercase it (Unpaywall normalises DOIs to
    lowercase and echoes them back lowercased on ``doi``).
    """
    key = identifier.strip()
    low = key.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key.lower()


class UnpaywallSource(PaperSource):
    """PaperSource implementation backed by the Unpaywall v2 REST API.

    ``identifier`` is a DOI (bare ``10.x/…``). Unpaywall is keyed on DOI only —
    there is no arXiv-id, title, or OpenAlex-id lookup, so seeds that carry no
    DOI never reach this source (the ``{"any_of": ["doi"]}`` dispatch filter).

    **No batch endpoint.** Unpaywall exposes only ``GET /v2/{doi}`` — one DOI per
    request — so ``get_papers`` iterates serially while keeping the
    ``list -> dict`` contract. There is no multi-id or filter endpoint to chunk.

    A missing DOI (HTTP 404) is terminal -> ``None``. Transient HTTP (429 rate
    limit, 5xx) raises so the worker leaves the message unarchived and pgmq
    redelivers after the visibility timeout — no in-process backoff, which would
    risk overrunning ``worker.VT_SECONDS``.
    """

    def __init__(self, email: str | None = None) -> None:
        # A valid email is mandatory. Prefer an explicit arg (future settings
        # wiring), else the UNPAYWALL_EMAIL env var. Resolution is deferred to
        # get_papers so a missing credential surfaces as a clear runtime error
        # at fetch time rather than at import/construction.
        self._email = email or os.environ.get(_EMAIL_ENV) or None
        self._session = requests.Session()

    def get_papers(self, identifiers: list[str]) -> dict[str, UnpaywallRecord | None]:
        """Serial-fetch by DOI; missing DOIs map to None (terminal, don't retry).

        No native batch endpoint exists, so this loops one DOI at a time. HTTP
        404 (unknown DOI) records None; transient 429/5xx raise so the worker
        leaves the messages unarchived for pgmq redelivery. Retrying in-process
        instead would risk blocking past ``worker.VT_SECONDS``, which lets pgmq
        hand the same batch to a second worker while this one is still sleeping.
        """
        out: dict[str, UnpaywallRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        if not self._email:
            raise RuntimeError(
                f"{_EMAIL_ENV} is required: Unpaywall rejects every request without a "
                "valid email query param. Set it in the environment or v3/harvest/.env."
            )

        logger.info("fetching metadata for %s", identifiers)
        for req in identifiers:
            doi = _normalize_doi(req)
            resp = self._session.get(
                f"{_BASE_URL}{doi}",
                params={"email": self._email},
                timeout=_TIMEOUT_SECONDS,
            )
            if resp.status_code == 404:
                logger.warning("no result for %s (not found)", req)
                continue
            if resp.status_code != 200:
                # 429 (rate limit) and 5xx are transient — raise so pgmq redelivers.
                logger.error("fetch failed (HTTP %s) for %s", resp.status_code, req)
                resp.raise_for_status()
                # Non-200, non-error status (e.g. an unexpected 3xx) — treat as
                # transient rather than silently record a miss.
                raise RuntimeError(f"unexpected HTTP {resp.status_code} for {req}")

            result = resp.json()
            if not result or not isinstance(result, dict):
                continue
            out[req] = UnpaywallRecord.from_result(result)

        return out
