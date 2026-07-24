import json
import logging
import os
from collections.abc import Iterator

import requests

from .base import PaperSource
from .models.opencitations import OpenCitationsRecord

logger = logging.getLogger("harvest.opencitations")

_BASE_URL = "https://api.opencitations.net/index/v2"

# OpenCitations' index is keyed on OMID but resolves a `doi:`-prefixed DOI. A BARE
# {doi} path 404s — the `doi:` prefix is REQUIRED.
_ID_PREFIX = "doi:"

# Optional access token. OpenCitations needs NO credential — every endpoint works
# unauthenticated — but a token raises the rate limit when present.
_TOKEN_ENV = "OPENCITATIONS_ACCESS_TOKEN"

# Per-request timeout: (connect, read) seconds. SHORT read on purpose: a slow body
# is capped, not waited out. OpenCitations returns the WHOLE edge array in one
# response (there is no working server-side pagination — `limit`/`offset`/count
# params bust its cache and hang, probed live), so a well-cited DOI either streams
# fast (AlexNet: 28k citations, ~2s) or the server can't build it at all and stalls
# past the read timeout (ResNet/CVPR: 503s only after ~70s). The 30s read timeout
# lets that pathological case fail fast (raised -> pgmq redelivers -> poison-pill
# backstop) instead of hanging the whole batch for minutes.
_TIMEOUT_SECONDS = (10.0, 30.0)

# Hard cap on edges pulled per direction, per DOI. A single popular DOI streams
# tens of thousands of edges (AlexNet: 28,448 citations, ~9.6 MB — right at the
# worker's 10 MB max_response_size). We DON'T need the full graph: cap the stream,
# stop reading, and persist what we have. 2,000 edges/direction (~0.7 MB) keeps a
# single DOI's fetch well under the read timeout and its write well under the size
# guard, while still capturing a substantial citation neighborhood.
_MAX_EDGES_PER_DIRECTION = 2000

# Bytes per streamed chunk while incrementally parsing the array.
_STREAM_CHUNK_BYTES = 1 << 16


def _normalize_doi(identifier: str) -> str:
    """A seed carries the DOI bare (``10.x/…``); strip any URL/``doi:`` wrapper, lowercase.

    OpenCitations returns DOIs lowercased and its ``doi:`` id form is
    case-insensitive, so normalise to bare-lowercased and let the source add the
    required ``doi:`` prefix.
    """
    key = identifier.strip()
    low = key.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key.lower()


class OpenCitationsSource(PaperSource):
    """PaperSource backed by the OpenCitations Index v2 REST API.

    ``identifier`` is a DOI (bare ``10.x/…``). OpenCitations' index is
    DOI-addressable via the ``doi:`` id form — there is no arXiv-id/title lookup —
    so seeds without a DOI never reach this source (the ``{"any_of": ["doi"]}``
    dispatch filter).

    **No batch endpoint.** The API exposes only per-id path lookups
    (``GET /index/v2/references/doi:{doi}`` and ``/index/v2/citations/doi:{doi}``)
    — one DOI per request, per direction — so ``get_papers`` iterates serially
    while keeping the ``list -> dict`` contract. There is no multi-id or filter
    endpoint to chunk.

    A missing DOI (HTTP 404) is terminal -> None. A DOI present in the index but
    with no refs/citations returns HTTP 200 with an empty array — a valid record
    with zero edges, NOT a miss. Transient HTTP (429 rate limit, 5xx) raises so the
    worker leaves the message unarchived and pgmq redelivers after the visibility
    timeout — no in-process backoff, which would risk overrunning
    ``worker.VT_SECONDS``.

    **Bounded fetch.** Each direction's response is streamed and truncated at
    ``_MAX_EDGES_PER_DIRECTION`` edges: a heavily-cited DOI (tens of thousands of
    edges) can't produce an unbounded fetch that blows the batch timeout. When the
    cap is hit we stop reading and persist the partial neighborhood — this is a
    ``succeeded`` record, not a failure. Because a batch of these runs at
    ``batch_size=1`` (see ``worker.SOURCES``), a DOI the server simply can't serve
    (its stream stalls past the read timeout) poisons only its own message, never a
    batch of faster neighbors.

    No API key required. An optional ``OPENCITATIONS_ACCESS_TOKEN`` raises the rate
    limit when present, but every endpoint works unauthenticated.
    """

    def __init__(self, access_token: str | None = None) -> None:
        self._session = requests.Session()
        token = access_token or os.environ.get(_TOKEN_ENV) or None
        if token:
            self._session.headers["authorization"] = token

    def get_papers(self, identifiers: list[str]) -> dict[str, OpenCitationsRecord | None]:
        """Serial-fetch by DOI; missing DOIs map to None (terminal, don't retry).

        No native batch endpoint exists, so this loops one DOI at a time, hitting
        both the references and citations endpoints per DOI. HTTP 404 (DOI not in
        the index) records None; transient 429/5xx raise so the worker leaves the
        messages unarchived for pgmq redelivery. Retrying in-process instead would
        risk blocking past ``worker.VT_SECONDS``, which lets pgmq hand the same
        batch to a second worker while this one is still sleeping.
        """
        out: dict[str, OpenCitationsRecord | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching citations for %s", identifiers)
        for req in identifiers:
            doi = _normalize_doi(req)
            references = self._fetch("references", doi, req)
            if references is None:
                # DOI not resolvable in the index (both directions 404 together).
                logger.warning("no result for %s (not found)", req)
                continue
            citations = self._fetch("citations", doi, req)
            if citations is None:
                citations = []
            out[req] = OpenCitationsRecord.from_responses(doi, references, citations)

        return out

    def _fetch(self, direction: str, doi: str, req: str) -> list[dict[str, object]] | None:
        """GET one direction for one DOI. 404 -> None (miss); transient -> raise.

        A 200 with an empty array is a real (edge-less) result, distinct from the
        404 miss that returns None. The body is streamed and truncated at
        ``_MAX_EDGES_PER_DIRECTION`` so one popular DOI can't produce an unbounded
        fetch; hitting the cap is logged but is NOT a failure — the partial edge set
        is returned and persisted.
        """
        resp = self._session.get(
            f"{_BASE_URL}/{direction}/{_ID_PREFIX}{doi}",
            timeout=_TIMEOUT_SECONDS,
            stream=True,
        )
        try:
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                # 429 (rate limit) and 5xx are transient — raise so pgmq redelivers.
                logger.error("fetch failed (HTTP %s) for %s %s", resp.status_code, direction, req)
                resp.raise_for_status()
                # Non-200, non-error status (e.g. an unexpected 3xx) — treat as
                # transient rather than silently record a miss.
                raise RuntimeError(f"unexpected HTTP {resp.status_code} for {direction} {req}")

            edges: list[dict[str, object]] = []
            for obj in _iter_json_array(resp, _STREAM_CHUNK_BYTES):
                edges.append(obj)
                if len(edges) >= _MAX_EDGES_PER_DIRECTION:
                    logger.warning(
                        "edge cap (%d) hit for %s %s — persisting partial neighborhood",
                        _MAX_EDGES_PER_DIRECTION,
                        direction,
                        req,
                    )
                    break
            return edges
        finally:
            # Stop reading and release the connection — critical when we broke out
            # early at the cap so the rest of a huge body is never downloaded.
            resp.close()


def _iter_json_array(resp: requests.Response, chunk_bytes: int) -> Iterator[dict[str, object]]:
    """Incrementally yield the top-level objects of a streamed JSON array.

    OpenCitations returns a bare JSON array of edge objects. Parsing it whole would
    buffer the entire (multi-MB) body before the caller can apply its edge cap, so
    this decodes object-by-object off the stream and lets the caller stop early.
    Dependency-free: a growing text buffer fed to ``json.JSONDecoder.raw_decode``.
    Non-array bodies (e.g. an error object) yield nothing.
    """
    decoder = json.JSONDecoder()
    buf = ""
    opened = False
    for chunk in resp.iter_content(chunk_size=chunk_bytes, decode_unicode=True):
        if not chunk:
            continue
        buf += chunk
        if not opened:
            start = buf.find("[")
            if start == -1:
                continue
            buf = buf[start + 1 :]
            opened = True
        # Decode as many complete objects as the buffer currently holds.
        while True:
            stripped = buf.lstrip().lstrip(",").lstrip()
            if not stripped:
                buf = stripped
                break
            if stripped[0] == "]":
                return
            try:
                obj, end = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                # Partial object at the buffer tail — wait for more bytes.
                buf = stripped
                break
            if isinstance(obj, dict):
                yield obj
            buf = stripped[end:]
