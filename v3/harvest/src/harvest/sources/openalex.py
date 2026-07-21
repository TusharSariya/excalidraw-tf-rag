import logging

import pyalex
from pyalex import Works
from requests.exceptions import HTTPError

from .models.openalex import OpenAlexWork

logger = logging.getLogger("harvest.openalex")

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


def _works_key(identifier: str) -> str:
    """Normalise a seed identity to a form OpenAlex's ``/works/<key>`` resolves.

    ``Works()[key]`` hits ``/works/<key>`` verbatim. OpenAlex resolves an OpenAlex id
    (``W…``, or its ``https://openalex.org/W…`` URL), a DOI in URL form
    (``https://doi.org/10.…``), and namespaced external ids (``pmid:``, ``pmcid:``,
    ``mag:``) — but NOT a bare ``10.x`` DOI, which 404s. Seeds carry the DOI bare, so
    wrap it; pass anything already resolvable through untouched.

    ROR is normalised but **inert**: it identifies an *institution*, so
    ``/works/ror:<id>`` 404s — there is no work with a ROR identity. Seeds carry no ROR
    today; it is normalised here for completeness (one place for a future institutions
    path to hook), but it will not resolve a paper via ``get_paper``.
    """
    key = identifier.strip()
    if key.startswith("10."):
        return f"https://doi.org/{key}"
    low = key.lower()
    if low.startswith("ror:"):
        return key
    if "ror.org/" in low:
        return f"ror:{key.rsplit('/', 1)[-1]}"
    return key


class OpenAlexSource:
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
        # Retry the codes worth retrying. pyalex defaults max_retries=0, i.e. a single
        # 429 propagates as a hard failure — unacceptable for a queue worker, since a
        # transient rate-limit would look identical to a genuine fetch error.
        pyalex.config.max_retries = 3
        pyalex.config.retry_backoff_factor = 0.5
        pyalex.config.retry_http_codes = [429, 500, 502, 503, 504]

    def get_paper(self, identifier: str) -> OpenAlexWork | None:
        return self._fetch(identifier)

    def _fetch(self, identifier: str) -> OpenAlexWork | None:
        logger.info("fetching metadata for %s", identifier)
        try:
            result = Works()[_works_key(identifier)]
        except HTTPError as e:
            # pyalex surfaces everything as requests' HTTPError, so separate "this
            # paper isn't in OpenAlex" (terminal: not_found) from "OpenAlex is having
            # a bad day" (transient: retry). Collapsing them would either poison-pill
            # good seeds or spin forever on genuinely absent ones.
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                logger.warning("no result for %s (not found)", identifier)
                return None
            logger.error("fetch failed (HTTP %s): %s", status, e)
            raise
        return OpenAlexWork.from_result(result)
