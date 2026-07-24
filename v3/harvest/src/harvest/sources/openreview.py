import logging

import requests

from .base import PaperSource
from .models.openreview import OpenReviewNote

logger = logging.getLogger("harvest.openreview")

# The api2 REST surface. OpenReview reads are anonymous — no credentials/API key
# are required (contrast OpenAlex). We hit the plain JSON endpoint directly with
# `requests` rather than the openreview-py SDK: the SDK pulls a heavy transitive
# tree (lxml, cryptography, pycryptodome, and the Cython C-extension
# `editdistance`, which has no cp314 wheel and fails to compile in the slim uv
# builder image). `GET /notes?id=<id>` is exactly what the SDK's get_note wraps.
OPENREVIEW_NOTES_URL = "https://api2.openreview.net/notes"

# A single request's timeout. Sustained rate limits / 5xx / the challenge gate are
# left to pgmq redelivery (worker.VT_SECONDS), not an in-process sleep.
REQUEST_TIMEOUT = 30


class OpenReviewSource(PaperSource):
    """PaperSource implementation backed by OpenReview's api2 REST endpoint.

    ``identifier`` is an OpenReview note/forum id (``seeds.openreview_id``), sent
    to ``GET /notes?id=<id>`` verbatim. For a submission the note id equals its
    ``forum`` id, which is the identity seeds carry. Identity is never resolved by
    fuzzy title/venue search — an unknown id maps to None (terminal not-found), the
    same discipline the OpenAlex adapter documents.

    api2 has no true multi-id batch endpoint, so this is a serial per-id loop that
    keeps the ``get_papers(list) -> dict`` contract.

    NOTE: api2.openreview.net sits behind an anti-bot challenge gate that returns
    HTTP 403 (``ChallengeRequiredError``) to non-allow-listed egress IPs. That is a
    transient, environment-level failure, not a missing record — the request
    raises, the exception propagates, and pgmq redelivers the batch (possibly from
    a different egress) after the visibility timeout. It is deliberately NOT
    swallowed as not-found; only a real 404 / empty result is terminal.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "harvest/openreview"})

    def get_papers(self, identifiers: list[str]) -> dict[str, OpenReviewNote | None]:
        """Fetch notes by id; missing ids map to None (terminal, don't retry).

        * a genuinely missing note (HTTP 404, or a 200 with an empty ``notes``
          list) maps to None — terminal, do not retry;
        * everything else (the 403 challenge gate, 429 rate limits, 5xx) re-raises
          so the worker leaves the message unarchived for pgmq redelivery.
        """
        out: dict[str, OpenReviewNote | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching notes for %s", identifiers)

        for fid in identifiers:
            resp = self._session.get(
                OPENREVIEW_NOTES_URL, params={"id": fid}, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 404:
                logger.warning("no openreview note for %s (not found)", fid)
                continue
            # 403 challenge gate / 429 / 5xx -> raise for pgmq redelivery.
            resp.raise_for_status()

            notes = (resp.json() or {}).get("notes") or []
            if not notes:
                logger.warning("no openreview note for %s (empty result)", fid)
                continue

            out[fid] = OpenReviewNote.from_dict(notes[0])

        return out
