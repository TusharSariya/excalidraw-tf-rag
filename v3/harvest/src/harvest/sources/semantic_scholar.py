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


class SemanticScholarSource(PaperSource):
    """PaperSource implementation backed by the Semantic Scholar Graph API.

    ``identifier`` is a Semantic Scholar external paper id, already namespaced
    by the caller (e.g. "ARXIV:1706.03762" or "DOI:10.1145/..."), and passed to
    the Graph API verbatim.
    """

    def __init__(self) -> None:
        self._client = SemanticScholar()

    def get_papers(self, identifiers: list[str]) -> dict[str, S2Record | None]:
        """Batch-fetch papers; missing ids map to None (terminal, don't retry).

        Uses ``POST /graph/v1/paper/batch`` via ``get_papers``. Passes
        ``Paper.FIELDS`` explicitly — the SDK's batch default is
        ``Paper.SEARCH_FIELDS``, which omits ``citations`` / ``references`` /
        ``embedding`` / ``tldr`` (needed for ``s2_citation``, ``s2_reference``,
        ``s2_embedding``). Only a genuine not-found is recorded as None. Rate
        limits / 5xx propagate so the worker leaves the message unarchived for
        pgmq redelivery.
        """
        out: dict[str, S2Record | None] = {i: None for i in identifiers}
        if not identifiers:
            return out

        logger.info("fetching papers for %s", identifiers)
        remaining = set(identifiers)

        try:
            papers, missing = self._client.get_papers(
                paper_ids=list(identifiers),
                fields=Paper.FIELDS,
                return_not_found=True,
            )
        except ObjectNotFoundException:
            logger.warning("no result for %s (not found)", identifiers)
            return out
        except BadQueryParametersException:
            # Client raises this when *every* id in the batch is invalid/missing
            # ("No valid paper ids given") — treat as all not-found.
            logger.warning("no valid papers in batch %s", identifiers)
            return out

        for pid in missing:
            out[pid] = None
            remaining.discard(pid)

        for paper in papers:
            key = requested_id_for(paper, remaining)
            if key is None:
                logger.warning(
                    "batch paper paperId=%s matched no outstanding request id",
                    paper.paperId,
                )
                continue
            out[key] = S2Record.from_sdk(paper)
            remaining.discard(key)

        for pid in remaining:
            out[pid] = None

        return out
