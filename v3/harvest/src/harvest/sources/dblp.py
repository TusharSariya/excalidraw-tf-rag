import logging

import requests

from .base import PaperSource
from .models.dblp import DblpRecord

logger = logging.getLogger("harvest.dblp")

# dblp's record API is one key per request: GET https://dblp.org/rec/<key>.xml.
# There is NO batch / multi-id endpoint (the record export is XML-only; the .json
# variant 404s), so get_papers loops serially while keeping the list->dict contract.
_RECORD_URL = "https://dblp.org/rec/{key}.xml"

# dblp asks API consumers to identify themselves via User-Agent (no API key exists).
_USER_AGENT = "excalidraw-tf-rag-harvester/1.0 (+https://github.com/TusharSariyaOrg)"

_TIMEOUT_SECONDS = 30


class DblpSource(PaperSource):
    """PaperSource implementation backed by dblp.org's per-record XML API.

    ``identifier`` is a dblp record key (``conf/nips/KrizhevskySH12``,
    ``journals/corr/VaswaniSPUJGKP17``) — the value carried on ``seeds.dblp_id``.
    dblp needs no API key, so there is no credential to be missing; the source is
    always constructible.

    There is no native batch/multi-id endpoint, so ``get_papers`` fetches serially,
    one HTTP GET per key, but still honours the ``list -> dict`` contract keyed by
    the requested id. Missing records (HTTP 404) map to ``None`` (terminal; the
    worker marks not_found). Transient HTTP (429/5xx) and any other non-2xx RAISE
    so the worker leaves the messages unarchived and pgmq redelivers them after the
    visibility timeout — retrying in-process would risk blocking past worker.VT.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def get_papers(self, identifiers: list[str]) -> dict[str, DblpRecord | None]:
        out: dict[str, DblpRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching records for %s", identifiers)
        for dblp_id in identifiers:
            url = _RECORD_URL.format(key=dblp_id)
            resp = self._session.get(url, timeout=_TIMEOUT_SECONDS)
            if resp.status_code == 404:
                logger.warning("no result for %s (not found)", dblp_id)
                continue
            # 429/5xx and any other non-2xx raise -> unarchived -> pgmq redelivers.
            resp.raise_for_status()

            record = DblpRecord.from_xml(dblp_id, resp.text)
            if record is None:
                logger.warning("could not parse dblp record for %s", dblp_id)
                continue
            out[dblp_id] = record
        return out
