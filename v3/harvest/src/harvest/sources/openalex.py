import logging
import os
from dataclasses import dataclass

import pyalex
from pyalex import Works
from requests.exceptions import HTTPError

from .base import Author, Paper

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
        # pyalex configures globally, not per-client. The key is OPTIONAL — the API
        # answers without one — but it raises the free daily allowance from $0.10 to
        # $1 (roughly 1k -> 10k list/filter calls), so a worker fleet wants it.
        key = api_key or os.environ.get("OPENALEX_API_KEY")
        if key:
            pyalex.config.api_key = key
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

    def get_paper(self, identifier: str) -> Paper | None:
        work = self._fetch(identifier)
        if work is None:
            return None
        return Paper(
            title=work.title,
            doi=work.doi,
            openalex_id=work.openalex_id,
            pdf_url=work.best_oa_pdf_url,
        )

    def get_authors(self, identifier: str) -> list[Author]:
        work = self._fetch(identifier)
        if work is None:
            return []
        authors: list[Author] = []
        for authorship in work.authorships:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if not name:
                continue
            authors.append(Author(name=name, author_id=author.get("id")))
        return authors

    def get_references(self, identifier: str) -> list[Paper]:
        """Unlike arXiv, OpenAlex has a citation graph — these are really populated.

        Returns FEWER papers than `work.referenced_works` has ids, sometimes by a lot
        (54 -> 44 on a sample record). The `openalex:W1|W2|...` filter is an OR and
        silently omits ids it cannot resolve — merged or deleted works — rather than
        erroring. So treat this as "the references OpenAlex can still resolve", not a
        complete list, and do not assert len(get_references()) == len(referenced_works).
        The raw ids survive on `OpenAlexWork.referenced_works` if you need the full set.
        """
        work = self._fetch(identifier)
        if work is None:
            return []
        if not work.referenced_works:
            return []

        # referenced_works is a list of ids only, so titles need a second call. Batch
        # them rather than fetching one work per reference.
        papers: list[Paper] = []
        ids = [_short_id(r) for r in work.referenced_works]
        for start in range(0, len(ids), REFERENCE_BATCH):
            batch = ids[start : start + REFERENCE_BATCH]
            logger.info("resolving %d reference(s) for %s", len(batch), identifier)
            try:
                results = Works().filter(openalex="|".join(batch)).get(per_page=REFERENCE_BATCH)
            except HTTPError as e:
                logger.error("reference batch failed: %s", e)
                continue
            papers.extend(self._to_paper(r) for r in results)
        return papers

    def get_citations(self, identifier: str) -> list[Paper]:
        work = self._fetch(identifier)
        if work is None:
            return []
        logger.info("fetching citations for %s", work.openalex_id)
        try:
            # `cited_by_api_url` is not always populated, so filter on cites=<id>
            # rather than trusting that field to exist.
            results = (
                Works().filter(cites=_short_id(work.openalex_id)).get(per_page=CITATIONS_PER_PAGE)
            )
        except HTTPError as e:
            logger.error("citation fetch failed: %s", e)
            return []
        return [self._to_paper(r) for r in results]

    def get_pdf_url(self, identifier: str) -> str | None:
        work = self._fetch(identifier)
        return work.best_oa_pdf_url if work else None

    @staticmethod
    def _to_paper(result: dict) -> Paper:
        return Paper(
            title=result.get("title") or "",
            doi=result.get("doi"),
            openalex_id=result.get("id"),
            pdf_url=(result.get("best_oa_location") or {}).get("pdf_url"),
        )

    def _fetch(self, identifier: str) -> "OpenAlexWork | None":
        logger.info("fetching metadata for %s", identifier)
        try:
            result = Works()[identifier]
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


@dataclass
class OpenAlexWork:
    """The subset of an OpenAlex Work that `public.source_openalex` stores.

    Mirrors ArxivPaper's shape: keep the useful fields typed, and hold onto `raw` so
    the upsert can persist the whole record without this class having to grow a field
    every time OpenAlex adds one.
    """

    openalex_id: str
    title: str
    doi: str | None
    publication_date: str | None
    type: str | None
    best_oa_pdf_url: str | None
    cited_by_count: int | None
    authorships: list[dict]
    referenced_works: list[str]
    raw: dict

    @classmethod
    def from_result(cls, result: dict) -> "OpenAlexWork":
        return cls(
            openalex_id=result.get("id") or "",
            title=result.get("title") or "",
            doi=result.get("doi"),
            publication_date=result.get("publication_date"),
            type=result.get("type"),
            best_oa_pdf_url=(result.get("best_oa_location") or {}).get("pdf_url"),
            cited_by_count=result.get("cited_by_count"),
            authorships=result.get("authorships") or [],
            referenced_works=result.get("referenced_works") or [],
            # dict(result) — pyalex returns a Work, a dict subclass. Store a plain
            # dict so json serialisation for the `raw` jsonb column is predictable.
            raw=dict(result),
        )
