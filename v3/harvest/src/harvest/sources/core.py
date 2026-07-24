import logging

import requests

from .base import PaperSource
from .models.core import CoreWork

logger = logging.getLogger("harvest.core")

# CORE v3 has no native multi-id / multi-DOI lookup — every request resolves a
# single DOI. get_papers therefore runs a serial loop (one HTTP call per id) while
# still honouring the list->dict contract. See the class docstring.
CORE_API_BASE = "https://api.core.ac.uk/v3"

# Per-request network timeout (connect, read) in seconds. CORE is occasionally slow
# on cold fulltext lookups; keep the read budget generous but bounded so a hung
# socket cannot outlive the worker's visibility timeout.
_TIMEOUT = (10, 30)


class CoreApiKeyMissing(RuntimeError):
    """Raised by ``get_papers`` when ``CORE_API_KEY`` is required but absent.

    The CORE v3 API is bearer-auth only — there is no unauthenticated allowance —
    so without a key the source cannot make a single call. The worker/loop treats
    this as *skipped* (not failed): a config gap, not a transient fetch error.
    """


def _match_doi(result: dict, requested_doi: str) -> bool:
    """Does a CORE work result actually carry the DOI we asked for?

    CORE's ``q=doi:"…"`` search is a filter, not an exact-get, so guard against a
    fuzzy match sneaking the wrong artefact in — persist only on a real DOI hit.
    Comparison is case-insensitive; DOIs are case-insensitive by spec.
    """
    got = result.get("doi")
    return bool(got) and got.strip().lower() == requested_doi.strip().lower()


class CoreSource(PaperSource):
    """PaperSource implementation backed by the CORE v3 API.

    ``identifier`` is a bare DOI (``10.…``). CORE resolves a work by DOI via its
    ``GET /v3/search/works`` endpoint (``q=doi:"<doi>"``, ``limit=1``); the numeric
    work ``id`` it returns doubles as the OA fulltext download id, and
    ``downloadUrl`` / ``fullTextIdentifier`` carry the OA fulltext URL.

    **Bearer auth is mandatory.** CORE v3 has no unauthenticated tier, so a missing
    ``CORE_API_KEY`` cannot yield even a degraded result — ``get_papers`` raises
    ``CoreApiKeyMissing`` and the caller reports the source *skipped*, distinct from
    a *not_found* (DOI genuinely absent) or an *error* (transient fetch failure).

    **No native batch endpoint.** CORE resolves one DOI per request, so
    ``get_papers`` loops serially — but the list->dict, key-by-requested-id contract
    is identical to the batch sources.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._session = requests.Session()

    def get_papers(self, identifiers: list[str]) -> dict[str, CoreWork | None]:
        """Serial-fetch by DOI; missing DOIs map to None (terminal, don't retry).

        A missing record (no hit, or an HTTP 404) is None — terminal, the worker
        marks it not_found. Transient HTTP (429 / 5xx) and network errors RAISE, so
        the worker leaves the messages unarchived and pgmq redelivers them after the
        visibility timeout. Nothing sleeps/back-offs in-process.

        Raises ``CoreApiKeyMissing`` up front when no key is configured — the whole
        batch is skipped rather than each id failing individually.
        """
        out: dict[str, CoreWork | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        if not self._api_key:
            raise CoreApiKeyMissing(
                "CORE_API_KEY is required (CORE v3 is bearer-auth only) but is not set "
                "in the environment or .env — the core source is skipped."
            )

        logger.info("fetching works for %s", identifiers)
        for doi in identifiers:
            work = self._fetch_one(doi)
            if work is not None:
                out[doi] = work
        return out

    def _fetch_one(self, doi: str) -> CoreWork | None:
        """Resolve one DOI to a ``CoreWork`` (None if CORE has no matching work)."""
        resp = self._session.get(
            f"{CORE_API_BASE}/search/works",
            headers={"Authorization": f"Bearer {self._api_key}"},
            params={"q": f'doi:"{doi}"', "limit": 1},
            timeout=_TIMEOUT,
        )

        if resp.status_code == 404:
            logger.warning("no result for %s (not found)", doi)
            return None
        if resp.status_code != 200:
            # 429 / 5xx (and anything else unexpected) -> raise so pgmq redelivers.
            logger.error(
                "fetch failed (HTTP %s) for %s: %s", resp.status_code, doi, resp.text[:200]
            )
            resp.raise_for_status()
            # raise_for_status only raises on >=400; guard the <400-non-200 case.
            raise requests.HTTPError(f"unexpected HTTP {resp.status_code} for {doi}")

        results = (resp.json() or {}).get("results") or []
        for result in results:
            if isinstance(result, dict) and _match_doi(result, doi):
                return CoreWork.from_result(result)

        logger.warning("no result for %s (not found)", doi)
        return None
