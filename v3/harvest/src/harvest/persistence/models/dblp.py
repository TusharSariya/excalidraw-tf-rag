"""Table-shaped DTO for public.source_dblp + DblpRecord -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's
bigint ``paper_id``. The typed columns are the queryable subset; the whole XML
response is kept verbatim on ``raw`` (``{"xml": <text>}``), with ``authors``
broken out to a jsonb column for convenience.

``DblpRecord`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.dblp import DblpRecord


class SourceDblpRow(TableRow):
    """Insert/upsert shape for public.source_dblp (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``authors`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable dblp fields. ``ee`` is the list of
    electronic-edition URLs; ``doi`` is derived from an ``ee`` on ``doi.org``.
    """

    __table__ = "source_dblp"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    dblp_id: str | None = None
    record_type: str | None = None
    title: str | None = None
    authors: list[dict[str, Any]] | None = None
    venue: str | None = None
    year: int | None = None
    ee: list[str] | None = None
    doi: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def dblp_write(
    seed_paper_id: int,
    record: DblpRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_dblp`` success ``Write`` for one fetched ``DblpRecord``.

    A single row group keyed by the seed's ``paper_id`` — dblp exposes no citation
    graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceDblpRow(
        paper_id=seed_paper_id,
        status="succeeded",
        dblp_id=record.dblp_id or None,
        record_type=record.record_type,
        title=record.title,
        authors=[{"name": a.name, "orcid": a.orcid} for a in record.authors] or None,
        venue=record.venue,
        year=record.year,
        ee=list(record.ee) or None,
        doi=record.doi,
        raw=record.raw or None,
        fetched_at=now,
    )
    return Write(groups=[[row]])
