import logging

import pyalex
from pyalex import Works
from requests.exceptions import HTTPError

from harvest.errors import RateLimited

from .base import PaperSource
from .models.openalex import OpenAlexWork

logger = logging.getLogger("harvest.openalex")


def _reraise_http(exc: HTTPError, *, what: str) -> None:
    """Map OpenAlex HTTP failures: 404 stays local; 429 → RateLimited; else re-raise."""
    status = exc.response.status_code if exc.response is not None else None
    if status == 429:
        retry_after = None
        if exc.response is not None:
            retry_after = exc.response.headers.get("Retry-After")
        logger.error("fetch failed (HTTP 429) for %s — stopping drain", what)
        raise RateLimited("openalex", retry_after=retry_after) from exc
    logger.error("fetch failed (HTTP %s): %s", status, exc)
    raise


# How many citing works to pull. OpenAlex pages at 25 by default; per_page maxes at
# 200. Mirrors the limit=100 SemanticScholarSource uses for citations.
CITATIONS_PER_PAGE = 100

# Batch size for resolving referenced_works (which come back as bare OpenAlex IDs)
# into titles. The `openalex:W1|W2|...` filter is an OR over ids, so this turns N
# lookups into ceil(N/50) — and 50 keeps the request URL well inside limits.
REFERENCE_BATCH = 50


def _short_id(openalex_url: str) -> str:
    """https://openalex.org/W123 -> W123. The filter API wants bare ids."""
    return openalex_url.rsplit("/", 1)[-1]


def _normalize_key(identifier: str) -> str:
    """Normalise a seed identity to a form OpenAlex's ``/works/<key>`` resolves.

    ``Works()[key]`` hits ``/works/<key>`` verbatim. OpenAlex resolves an OpenAlex id
    (``W…``, or its ``https://openalex.org/W…`` URL), a DOI in URL form
    (``https://doi.org/10.…``), and namespaced external ids (``pmid:``, ``pmcid:``,
    ``mag:``) — but NOT a bare ``10.x`` DOI, which 404s. Seeds carry the DOI bare, so
    wrap it; pass anything already resolvable through untouched.

    ROR is normalised but **inert**: it identifies an *institution*, so
    ``/works/ror:<id>`` 404s — there is no work with a ROR identity. Seeds carry no ROR
    today; it is normalised here for completeness (one place for a future institutions
    path to hook), but it will not resolve a paper via ``get_papers``.
    """
    key = identifier.strip()
    if key.startswith("10."):
        return f"https://doi.org/{key}"
    low = key.lower()
    if "ror.org/" in low:
        return f"ror:{key.rsplit('/', 1)[-1]}"
    return key


def _is_openalex_work_id(key: str) -> bool:
    short = _short_id(key)
    return short.upper().startswith("W") and short[1:].isdigit()


def requested_id_for(result: dict, requested: set[str]) -> str | None:
    """Map a Work dict back to one of the ids we asked for."""
    short = _short_id(result["id"])
    ids = result.get("ids") or {}
    doi_url = ids.get("doi")
    bare_doi = doi_url.removeprefix("https://doi.org/") if doi_url else None

    candidates: set[str] = {short, result["id"]}
    if doi_url:
        candidates.add(doi_url)
    if bare_doi:
        candidates.add(bare_doi)

    for req in requested:
        norm = _normalize_key(req)
        if req in candidates or norm in candidates or _short_id(norm) == short:
            return req
    return None


class OpenAlexSource(PaperSource):
    """PaperSource implementation backed by OpenAlex, via the pyalex SDK.

    **`identifier` is a DOI or an OpenAlex work ID — NOT an arXiv id.** This differs
    from ArxivSource and SemanticScholarSource, which both take an arXiv id, and the
    difference is forced rather than chosen: OpenAlex has no arXiv filter. There is no
    `ids.arxiv`, and arXiv's own DOI (10.48550/arXiv.<id>) does not resolve either —
    `Works()["https://doi.org/10.48550/arXiv.1706.03762"]` returns 404.

    Resolving an arXiv id by title search is a trap, not a fallback. Searching
    "Attention Is All You Need" returns a `type: preprint` dated 2025 whose PDF is
    hosted on a content-scraping mirror — a plausible-looking record for the wrong
    artefact. A harvester would upsert that happily and you would never know. So this
    class refuses to guess: give it a DOI or an OpenAlex id, or it returns None.

    `public.seeds` carries `openalex_id` and `doi` columns precisely so a seed can be
    resolved per-source; the caller passes whichever identity it has.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            pyalex.config.api_key = api_key
        else:
            logger.warning(
                "no OpenAlex API key: running on the ~$0.10/day unauthenticated "
                "allowance. Set OPENALEX_API_KEY for 10x the quota."
            )
        # Retry server errors only. Do NOT retry 429 in-process: pyalex's backoff
        # can pin a worker for minutes past the VT, and the worker must exit the
        # drain on RateLimited so pgmq spaces retries (see harvest.errors).
        pyalex.config.max_retries = 3
        pyalex.config.retry_backoff_factor = 0.5
        pyalex.config.retry_http_codes = [500, 502, 503, 504]

    def get_papers(self, identifiers: list[str]) -> dict[str, OpenAlexWork | None]:
        return self._fetch(identifiers)

    def _fetch(self, identifiers: list[str]) -> dict[str, OpenAlexWork | None]:
        """Batch-fetch; missing ids map to None (terminal, don't retry).

        ``Works()[list]`` only filters by OpenAlex work id, so W… ids go through
        that path. DOIs and other keys are resolved one-at-a-time via
        ``Works()[key]`` (which accepts DOI URLs). Transient HTTP errors raise so
        the worker leaves messages unarchived for redelivery.
        """
        out: dict[str, OpenAlexWork | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching metadata for %s", identifiers)
        remaining = set(identifiers)

        oa_shorts: list[str] = []
        singles: list[str] = []
        for req in identifiers:
            norm = _normalize_key(req)
            if _is_openalex_work_id(norm):
                oa_shorts.append(_short_id(norm))
            else:
                singles.append(req)

        if oa_shorts:
            try:
                results = Works()[oa_shorts]
            except HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning("no result for openalex batch %s (not found)", oa_shorts)
                else:
                    _reraise_http(e, what=f"openalex batch {oa_shorts}")
            else:
                if results is None:
                    results = []
                elif isinstance(results, dict):
                    results = [results]
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    key = requested_id_for(result, remaining)
                    if key is None:
                        logger.warning(
                            "openalex work %s matched no outstanding request id",
                            result.get("id"),
                        )
                        continue
                    out[key] = OpenAlexWork.from_result(result)
                    remaining.discard(key)

        for req in singles:
            try:
                result = Works()[_normalize_key(req)]
            except HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning("no result for %s (not found)", req)
                    continue
                _reraise_http(e, what=req)
            if not result or not isinstance(result, dict):
                continue
            out[req] = OpenAlexWork.from_result(result)
            remaining.discard(req)

        return out
