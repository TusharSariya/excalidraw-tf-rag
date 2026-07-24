import logging

import requests

from .base import PaperSource
from .models.osf import OsfPreprint

logger = logging.getLogger("harvest.osf")

_BASE_URL = "https://api.osf.io/v2/preprints/"

# Per-request timeout (seconds). A single preprint lookup is a small GET; a slow
# response is treated as transient and raised so pgmq redelivers the batch.
_TIMEOUT_SECONDS = 30.0

# OSF public reads are anonymous — no API key. A polite User-Agent identifies the
# harvester without carrying any credential.
_USER_AGENT = "excalidraw-tf-rag-harvest/1.0 (+https://osf.io)"


def _guid_for(identifier: str) -> str | None:
    """Resolve a seed DOI to the OSF preprint guid ``/v2/preprints/{guid}/`` accepts.

    OSF's preprints endpoint does NOT support ``filter[doi]`` (it 400s: "Value
    'doi' is not a filterable field") and has no external-DOI lookup at all. The
    only keyed lookup is by OSF-native guid. So this maps the seed identity to a
    guid where it can, and returns None where it cannot (a non-OSF DOI is simply
    unresolvable here — the worker records not_found, never a fuzzy title guess).

    Resolvable forms:

    * an **OSF-minted DOI** — ``10.31234/osf.io/k3fr9_v1`` or its URL form
      ``https://doi.org/10.31234/osf.io/k3fr9_v1`` — whose guid is the segment
      after ``osf.io/``;
    * a **bare OSF guid** handed in directly (``k3fr9_v1``), which carries no
      ``/`` and does not look like a DOI.

    A non-OSF DOI (``10.1371/journal.pone.…``, a Zenodo/ACM DOI, …) has no OSF
    identity and returns None.
    """
    key = identifier.strip()
    low = key.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            key = key[len(prefix) :]
            low = key.lower()
            break

    marker = "osf.io/"
    if marker in low:
        guid = key[low.index(marker) + len(marker) :].strip("/")
        return guid or None

    # A bare OSF guid: no path separators and not a DOI (which starts with "10.").
    if "/" not in key and not key.startswith("10."):
        return key or None

    return None


class OsfSource(PaperSource):
    """PaperSource implementation backed by the OSF v2 REST API (JSON:API).

    ``identifier`` is a DOI (the seed's ``doi``). **OSF is keyed on its own guid,
    not on an external DOI** — ``GET /v2/preprints/?filter[doi]=…`` returns HTTP
    400 ("not a filterable field") and there is no external-DOI lookup on this
    endpoint. So this class resolves the DOI to an OSF guid where it can (an
    OSF-minted DOI ``10.31234/osf.io/<guid>``, or a bare guid) and treats any
    other DOI as unresolvable -> ``None``. It never resolves identity by title
    search: a plausible-looking wrong record persists as confidently as a right
    one.

    **No batch endpoint.** OSF exposes only ``GET /v2/preprints/{guid}/`` — one
    guid per request; the listing endpoint supports pagination/filter but no
    multi-id filter — so ``get_papers`` iterates serially while keeping the
    ``list -> dict`` contract.

    A missing preprint (HTTP 404) is terminal -> ``None``. Transient HTTP (429
    rate limit, 5xx) raises so the worker leaves the message unarchived and pgmq
    redelivers after the visibility timeout — no in-process backoff, which would
    risk overrunning ``worker.VT_SECONDS``.
    """

    def __init__(self) -> None:
        # OSF public reads need no credential (anonymous). A session reuses the
        # connection across the serial loop.
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def get_papers(self, identifiers: list[str]) -> dict[str, OsfPreprint | None]:
        """Serial-fetch by OSF guid; missing/unresolvable ids map to None (terminal).

        No native batch endpoint exists, so this loops one guid at a time. A DOI
        that does not resolve to an OSF guid, and HTTP 404 (unknown guid), both
        record None; transient 429/5xx raise so the worker leaves the messages
        unarchived for pgmq redelivery. Retrying in-process instead would risk
        blocking past ``worker.VT_SECONDS``, which lets pgmq hand the same batch
        to a second worker while this one is still sleeping.
        """
        out: dict[str, OsfPreprint | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)
        for req in identifiers:
            guid = _guid_for(req)
            if not guid:
                logger.warning("no OSF guid resolvable from %s (not found)", req)
                continue

            resp = self._session.get(
                f"{_BASE_URL}{guid}/",
                timeout=_TIMEOUT_SECONDS,
            )
            if resp.status_code == 404:
                logger.warning("no result for %s (guid=%s, not found)", req, guid)
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
            out[req] = OsfPreprint.from_result(result)

        return out
