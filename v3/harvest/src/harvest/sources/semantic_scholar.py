import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, Literal

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)

from .base import PaperSource
from .models.s2 import RELATED_PAPER_KEYS, RelatedPaper, S2Record

logger = logging.getLogger("harvest.semantic_scholar")

# externalIds keys → the namespace prefix the worker puts on seed ids (ARXIV:/DOI:).
_EXT_TO_NS = {
    "ArXiv": "ARXIV",
    "DOI": "DOI",
    "CorpusId": "CorpusId",
    "ACL": "ACL",
    "PubMed": "PMID",
    "PubMedCentral": "PMCID",
    "MAG": "MAG",
}

#: Subject metadata only (no embedded citation/reference lists). The Graph API
#: often returns ``referenceCount`` / ``citationCount`` with empty nested arrays
#: on ``/paper/batch``; cites/refs are loaded via dedicated paginated endpoints.
SUBJECT_FIELDS: list[str] = [
    *Paper.SEARCH_FIELDS,
    "tldr",
    "embedding",
]

#: Thin stub fields for ``/citations`` and ``/references`` pages (enough for
#: ``s2_paper`` stubs + author absorb + edges).
RELATED_FIELDS: list[str] = [
    "paperId",
    "corpusId",
    "externalIds",
    "title",
    "year",
    "venue",
    "url",
    "citationCount",
    "authors",
]

#: S2 Graph API max page size for citation/reference listings.
RELATED_PAGE_LIMIT = 1000

# Back-compat aliases (pre-0.15 subject ladder names → subject-only fields).
META_FIELDS = SUBJECT_FIELDS
RICH_FIELDS = SUBJECT_FIELDS
THIN_FIELDS = SUBJECT_FIELDS
BATCH_FIELDS = SUBJECT_FIELDS


def requested_id_for(paper: Paper, requested: set[str]) -> str | None:
    """Map a batch Paper back to one of the ids we asked for.

    ``POST /paper/batch`` does not echo the request string — only ``paperId`` and
    ``externalIds`` (e.g. ``{"ArXiv": "1706.03762"}``). Rebuild candidate forms
    and pick the first still-outstanding request that matches.
    """
    candidates: set[str] = set()
    if paper.paperId:
        candidates.add(paper.paperId)
    for key, value in (paper.externalIds or {}).items():
        if value is None:
            continue
        text = str(value)
        candidates.add(text)
        ns = _EXT_TO_NS.get(key, key.upper())
        candidates.add(f"{ns}:{text}")

    for req in requested:
        if req in candidates:
            return req
    return None


def _is_all_ids_invalid(exc: BadQueryParametersException) -> bool:
    """True only for the SDK's empty-batch-ids case — not field/size 400s."""
    msg = str(exc).lower()
    return "no valid paper" in msg or "invalid paper id" in msg


def _paper_dict_from_edge(item: Any) -> dict[str, Any] | None:
    """Extract the nested paper payload from a Citation/Reference SDK object."""
    paper = getattr(item, "paper", None)
    if paper is not None:
        raw = getattr(paper, "raw_data", None)
        if isinstance(raw, dict):
            return raw
    raw_edge = getattr(item, "raw_data", None)
    if not isinstance(raw_edge, dict):
        return None
    nested = raw_edge.get("citingPaper") or raw_edge.get("citedPaper")
    return nested if isinstance(nested, dict) else None


def related_paper_from_edge(item: Any) -> RelatedPaper | None:
    """Map an SDK Citation/Reference item to a RelatedPaper stub."""
    nested = _paper_dict_from_edge(item)
    if nested is None:
        return None
    filtered = {k: v for k, v in nested.items() if k in RELATED_PAPER_KEYS}
    return RelatedPaper.from_dict(filtered)


class SemanticScholarSource(PaperSource):
    """PaperSource implementation backed by the Semantic Scholar Graph API.

    ``identifier`` is a Semantic Scholar external paper id, already namespaced
    by the caller (e.g. "ARXIV:1706.03762" or "DOI:10.1145/..."), and passed to
    the Graph API verbatim.
    """

    def __init__(self) -> None:
        self._client = SemanticScholar()

    def get_papers(self, identifiers: list[str]) -> dict[str, S2Record | None]:
        """Fetch subject metadata, then paginate citations and references.

        Subject fetch uses :data:`SUBJECT_FIELDS` only (no embedded cite/ref
        lists — those arrays are unreliable on batch/paper payloads). Graph edges
        come from ``get_paper_citations`` / ``get_paper_references`` at
        ``limit=1000`` until exhausted.

        Missing ids map to None (terminal). Rate limits / 5xx propagate so the
        worker leaves messages unarchived for pgmq redelivery.
        """
        out: dict[str, S2Record | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        for identifier in identifiers:
            out[identifier] = self._get_one(identifier)
        return out

    def _get_one(self, identifier: str) -> S2Record | None:
        logger.info("fetching paper %s", identifier)
        subject = self._fetch_subject(identifier)
        if subject is None:
            return None

        s2_id = subject.paper_id
        if not s2_id:
            logger.warning("subject for %s has no paper_id — treating as not found", identifier)
            return None

        citations = list(self._iter_related("citations", s2_id))
        references = list(self._iter_related("references", s2_id))
        record = replace(subject, citations=citations, references=references)
        logger.info(
            "fetched %s: cites=%s refs=%s emb=%s",
            identifier,
            len(citations),
            len(references),
            record.embedding is not None,
        )
        return record

    def _fetch_subject(self, identifier: str) -> S2Record | None:
        try:
            papers, missing = self._client.get_papers(
                paper_ids=[identifier],
                fields=SUBJECT_FIELDS,
                return_not_found=True,
            )
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", identifier)
            return None
        except BadQueryParametersException as exc:
            if _is_all_ids_invalid(exc):
                logger.warning("no valid paper id %s", identifier)
                return None
            logger.error("S2 rejected %s (subject fields): %s", identifier, exc)
            raise

        if identifier in missing or not papers:
            return None

        paper = papers[0]
        key = requested_id_for(paper, {identifier})
        if key != identifier:
            logger.warning(
                "paper paperId=%s did not map back to requested id %s",
                paper.paperId,
                identifier,
            )
            return None

        return S2Record.from_sdk(paper)

    def _iter_related(
        self,
        kind: Literal["citations", "references"],
        paper_id: str,
    ) -> Iterator[RelatedPaper]:
        """Yield RelatedPaper stubs from paginated citation/reference endpoints."""
        if kind == "citations":
            fetch = self._client.get_paper_citations
        else:
            fetch = self._client.get_paper_references

        logger.info("paginating %s for %s (limit=%s)", kind, paper_id, RELATED_PAGE_LIMIT)
        try:
            pages = fetch(
                paper_id,
                fields=RELATED_FIELDS,
                limit=RELATED_PAGE_LIMIT,
            )
        except ObjectNotFoundException:
            logger.warning("%s endpoint 404 for %s — empty %s", kind, paper_id, kind)
            return

        count = 0
        try:
            for item in pages:
                related = related_paper_from_edge(item)
                if related is None or not related.paper_id:
                    continue
                count += 1
                yield related
        except TypeError as exc:
            # Under rate-limit / flaky responses the SDK can see ``data: null``
            # and blow up on ``for item in results['data']``. Treat as end of
            # stream so a partial graph can still persist; redelivery retries.
            logger.warning(
                "null/broken %s page for %s after %s items (%s) — stopping pagination",
                kind,
                paper_id,
                count,
                exc,
            )
            return

        logger.info("paginated %s for %s: %s items", kind, paper_id, count)
