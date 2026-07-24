import logging

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)

from .base import PaperSource
from .models.s2 import S2Record

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

#: RELATED_PAPER-shaped cite/ref subfields (no ``citations.authors.name`` etc. —
#: batch rejects those). Prefer this when the response fits.
_RELATED_SUBFIELDS: tuple[str, ...] = (
    "paperId",
    "corpusId",
    "externalIds",
    "title",
    "abstract",
    "venue",
    "year",
    "url",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "publicationDate",
    "publicationTypes",
    "publicationVenue",
    "journal",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "citationStyles",
    "authors",
)

#: Full graph payload for one paper. Works for most DOIs at ``n=1``; mega-cited
#: papers (e.g. Attention) 400 with "maximum size" — fall back to ``THIN_FIELDS``.
RICH_FIELDS: list[str] = [
    *Paper.SEARCH_FIELDS,
    "tldr",
    "embedding",
    *[f"citations.{s}" for s in _RELATED_SUBFIELDS],
    *[f"references.{s}" for s in _RELATED_SUBFIELDS],
]

#: Thin cite/ref stubs — enough for ``s2_citation`` / ``s2_reference`` edges +
#: author absorb. Survives Attention-scale citation lists at ``n=1``.
THIN_FIELDS: list[str] = [
    *Paper.SEARCH_FIELDS,
    "tldr",
    "embedding",
    "citations.paperId",
    "citations.title",
    "citations.year",
    "citations.venue",
    "citations.externalIds",
    "citations.corpusId",
    "citations.authors",
    "citations.citationCount",
    "citations.url",
    "references.paperId",
    "references.title",
    "references.year",
    "references.venue",
    "references.externalIds",
    "references.corpusId",
    "references.authors",
    "references.citationCount",
    "references.url",
]

#: Last-resort: subject metadata only (no citation graph). Still gets embedding.
META_FIELDS: list[str] = [
    *Paper.SEARCH_FIELDS,
    "tldr",
    "embedding",
]

# Back-compat alias used by tests / docs.
BATCH_FIELDS = RICH_FIELDS

_FIELD_LADDER: tuple[tuple[str, list[str]], ...] = (
    ("rich", RICH_FIELDS),
    ("thin", THIN_FIELDS),
    ("meta", META_FIELDS),
)


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


def _is_response_too_large(exc: BadQueryParametersException) -> bool:
    return "maximum size" in str(exc).lower()


class SemanticScholarSource(PaperSource):
    """PaperSource implementation backed by the Semantic Scholar Graph API.

    ``identifier`` is a Semantic Scholar external paper id, already namespaced
    by the caller (e.g. "ARXIV:1706.03762" or "DOI:10.1145/..."), and passed to
    the Graph API verbatim.
    """

    def __init__(self) -> None:
        self._client = SemanticScholar()

    def get_papers(self, identifiers: list[str]) -> dict[str, S2Record | None]:
        """Fetch papers one id at a time with a rich→thin→meta field ladder.

        Multi-id ``POST /paper/batch`` with citation lists is unreliable: S2
        returns HTTP 400 once the payload is too large, and even on 200 it often
        zeros out ``citations`` for all but one paper in the batch. Fetching
        serially at ``n=1`` preserves full cite/ref graphs. Per id we try
        :data:`RICH_FIELDS`, then :data:`THIN_FIELDS`, then :data:`META_FIELDS`
        when the API says the response would exceed maximum size.

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
        last_size_error: BadQueryParametersException | None = None

        for label, fields in _FIELD_LADDER:
            try:
                papers, missing = self._client.get_papers(
                    paper_ids=[identifier],
                    fields=fields,
                    return_not_found=True,
                )
            except ObjectNotFoundException:
                logger.warning("no result for %s (not found)", identifier)
                return None
            except BadQueryParametersException as exc:
                if _is_all_ids_invalid(exc):
                    logger.warning("no valid paper id %s", identifier)
                    return None
                if _is_response_too_large(exc):
                    logger.warning(
                        "S2 response too large for %s with %s fields; trying next rung",
                        identifier,
                        label,
                    )
                    last_size_error = exc
                    continue
                logger.error("S2 rejected %s (%s fields): %s", identifier, label, exc)
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

            record = S2Record.from_sdk(paper)
            logger.info(
                "fetched %s via %s fields: cites=%s refs=%s emb=%s",
                identifier,
                label,
                len(record.citations or []),
                len(record.references or []),
                record.embedding is not None,
            )
            return record

        # Every rung said "too large" — should be rare (meta is tiny). Propagate.
        assert last_size_error is not None
        raise last_size_error
