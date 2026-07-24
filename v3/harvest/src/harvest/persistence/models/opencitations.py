"""Table-shaped DTOs for public.opencitations_edge + source_opencitations.

Mirrors ``persistence.models.s2``: a fan-out source. One ``OpenCitationsRecord``
(both directions of the index for a seed DOI) becomes a group of
``opencitations_edge`` rows (the citation graph) followed by the per-seed
``source_opencitations`` outcome row, as one FK-ordered ``Write``.

The edge is a DOI->DOI fact keyed by ``oci`` (a stable, globally unique edge id),
so it is NOT tied to a single seed — two different seeds can surface the same edge
(one via its references, the other via its citations), and both upsert the same
``oci`` row. Each element's whole dict is kept on ``opencitations_edge.raw`` (the
home of the untyped ``citing``/``cited`` bundle ids), which is where the full API
response is preserved — hence the per-seed outcome row keeps ``raw`` None, exactly
like ``source_semantic_scholar`` leaves its heavy payload to the graph tables.

``OpenCitationsRecord`` is imported only for typing — the mapper reads its
attributes and never constructs one, so persistence keeps no runtime dependency on
``sources``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.opencitations import OpenCitationsRecord


class OpenCitationsEdgeRow(TableRow):
    """Insert/upsert shape for public.opencitations_edge (one DOI->DOI citation).

    Keyed by ``oci`` (OpenCitations' stable edge identifier). ``citing_doi`` cites
    ``cited_doi``; both are bare, lowercased DOIs parsed from the response bundles.
    ``raw`` keeps the original element (preserving the omid/openalex/pmid ids that
    the typed DOI columns drop).
    """

    __table__ = "opencitations_edge"
    __conflict__ = ("oci",)

    oci: str
    citing_doi: str
    cited_doi: str
    creation: str | None = None
    timespan: str | None = None
    journal_sc: bool | None = None
    author_sc: bool | None = None
    raw: dict[str, Any] | None = None
    fetched_at: datetime | None = None


class SourceOpenCitationsRow(TableRow):
    """Insert/upsert shape for public.source_opencitations (per-seed outcome).

    Keyed by the seed's bigint ``paper_id`` — a different namespace from the
    DOI-keyed ``opencitations_edge`` graph table. One row per seed records whether
    the OpenCitations fetch succeeded and how many edges each direction yielded.
    """

    __table__ = "source_opencitations"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    doi: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def opencitations_write(
    seed_paper_id: int,
    rec: OpenCitationsRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the OpenCitations persistence payload for one fetched record.

    The ``opencitations_edge`` rows (deduped by ``oci`` across both directions)
    followed by the per-seed ``source_opencitations`` outcome row, as one
    ``Write``. Edges carry no FK (a DOI->DOI fact), and the outcome row FKs only
    into ``seeds`` (already present), so the edge group leads and the outcome row
    trails.

    Edges whose needed side has no ``doi:`` token are skipped (per the API note):
    a DOI->DOI edge needs both endpoints. Self-loops (``citing_doi == cited_doi``)
    are dropped too — never a real citation, and letting one through would be a
    permanent write failure (poison pill).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)

    edges: dict[str, OpenCitationsEdgeRow] = {}
    for edge in (*rec.references, *rec.citations):
        if not edge.oci or not edge.citing_doi or not edge.cited_doi:
            continue
        if edge.citing_doi == edge.cited_doi:
            continue
        edges.setdefault(
            edge.oci,
            OpenCitationsEdgeRow(
                oci=edge.oci,
                citing_doi=edge.citing_doi,
                cited_doi=edge.cited_doi,
                creation=edge.creation,
                timespan=edge.timespan,
                journal_sc=edge.journal_self_citation,
                author_sc=edge.author_self_citation,
                raw=edge.raw,
                fetched_at=now,
            ),
        )

    outcome = SourceOpenCitationsRow(
        paper_id=seed_paper_id,
        status="succeeded",
        doi=rec.doi or None,
        reference_count=len(rec.references),
        citation_count=len(rec.citations),
        # Full fidelity lives on opencitations_edge.raw (per element); the outcome
        # row keeps raw None to avoid re-storing a multi-MB array per seed
        # (mirrors source_semantic_scholar).
        raw=None,
        fetched_at=now,
    )

    groups: list[Sequence[TableRow]] = []
    if edges:
        groups.append(list(edges.values()))
    groups.append([outcome])
    return Write(groups=groups)
