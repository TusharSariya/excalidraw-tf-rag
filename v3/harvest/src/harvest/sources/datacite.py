import logging
from urllib.parse import quote

import requests

from .base import PaperSource
from .models.datacite import DataCiteRecord

logger = logging.getLogger("harvest.datacite")

# DataCite public REST API — no key required. Single-DOI records live at
# `/dois/{doi}`; the DOI is placed verbatim in the path (URL-encoded so its `/`
# is not read as a path separator).
_BASE_URL = "https://api.datacite.org/dois"

# Politeness: a short connect/read timeout. A genuine slow response is a transient
# condition — the resulting error propagates and pgmq redelivers, rather than us
# blocking the worker past its visibility timeout.
_TIMEOUT_SECONDS = 30

# Transient HTTP: raise so pgmq redelivers after the 120s VT. Never sleep/backoff
# in-process (see the arxiv/openalex sources for the same reasoning).
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def requested_id_for(record: DataCiteRecord, requested: set[str]) -> str | None:
    """Map a fetched record back to one of the DOIs we asked for.

    DataCite normalises DOIs to lowercase, so match case-insensitively against the
    record's ``doi``. Seeds carry the DOI bare (``10.x/y``); that is exactly what
    the record echoes back in ``attributes.doi``.
    """
    got = record.doi.lower()
    for req in requested:
        if req.lower() == got:
            return req
    return None


class DataCiteSource(PaperSource):
    """PaperSource implementation backed by the DataCite public REST API.

    ``identifier`` is a bare DOI (``10.x/y``). DataCite registers **dataset and
    software** DOIs (Zenodo, Dryad, figshare, …); most journal/paper DOIs are
    registered at Crossref instead and legitimately 404 here — that is a valid
    ``not_found``, not an error (verified live: ``10.5281/zenodo.1212303`` -> 200,
    ``10.1371/journal.pone.0098679`` and ``10.1109/CVPR.2016.90`` -> 404).

    Identity is resolved ONLY by DOI. There is no title-search fallback — a
    fuzzy match would upsert a plausible-looking record for the wrong artefact
    (see the OpenAlexSource docstring for the same refusal).
    """

    def __init__(self, api_key: str | None = None) -> None:
        # DataCite's public API needs no key; the parameter mirrors the other
        # sources' factory signature so the worker wiring stays uniform.
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "excalidraw-tf-rag-harvest/1.0 (+https://github.com)"}
        )

    def get_papers(self, identifiers: list[str]) -> dict[str, DataCiteRecord | None]:
        """Resolve by DOI; missing DOIs map to None (terminal, don't retry).

        DataCite has no usable multi-DOI batch endpoint: the ``?query=doi:…``
        Lucene form rejects the ``/`` in a DOI (HTTP 400 ``token_mgr_error``).
        The reliable route is the single-record endpoint ``/dois/{doi}`` — the
        DOI URL-encoded into the path — so we loop it once per id.

        A genuine 404 means the DOI is not registered at DataCite (most journal
        DOIs live at Crossref) -> None, terminal. Transient HTTP (429/5xx) raises
        so the worker leaves the message unarchived for pgmq redelivery; other
        unexpected non-2xx also raise.
        """
        out: dict[str, DataCiteRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)

        for doi in identifiers:
            # `quote(safe="")` encodes the DOI's `/` (and `:`) so it stays a
            # single path segment rather than being read as `/dois/10.5281/…`.
            url = f"{_BASE_URL}/{quote(doi, safe='')}"
            resp = self._session.get(url, timeout=_TIMEOUT_SECONDS)

            if resp.status_code == 404:
                logger.warning("no result for %s (not found at DataCite)", doi)
                continue
            if resp.status_code in _TRANSIENT_STATUS:
                logger.error("fetch failed (HTTP %s) for %s", resp.status_code, doi)
                resp.raise_for_status()
            resp.raise_for_status()

            item = resp.json().get("data")
            if not isinstance(item, dict):
                logger.warning("datacite response for %s had no `data` object", doi)
                continue

            record = DataCiteRecord.from_result(item)
            # The endpoint echoes exactly the DOI we asked for, but match through
            # requested_id_for so DataCite's lowercasing can't drop the record.
            key = requested_id_for(record, {doi})
            if key is None:
                logger.warning(
                    "datacite record %s did not match requested doi %s",
                    record.doi,
                    doi,
                )
                continue
            out[key] = record

        return out
