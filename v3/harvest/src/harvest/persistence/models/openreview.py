"""Table-shaped DTO for public.source_openreview + OpenReviewNote -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's bigint
``paper_id``. The typed columns are the queryable subset (``openreview_id``,
``title``, ``abstract``, ``authors``, ``pdf``, ``venueid``); the whole note is kept
on ``raw``.

``OpenReviewNote`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.openreview import OpenReviewNote


class SourceOpenReviewRow(TableRow):
    """Insert/upsert shape for public.source_openreview (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``authors`` is a text[]; ``raw`` is jsonb;
    the scalar columns are the promoted, queryable OpenReview fields.
    """

    __table__ = "source_openreview"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    openreview_id: str | None = None
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = []
    pdf: str | None = None
    venueid: str | None = None
    forum: str | None = None
    number: int | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def openreview_write(
    seed_paper_id: int,
    note: OpenReviewNote,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_openreview`` success ``Write`` for one fetched note.

    A single row group keyed by the seed's ``paper_id`` — OpenReview notes have no
    citation graph to fan out (contrast ``s2_write``), so ``raw`` retains the rest.
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceOpenReviewRow(
        paper_id=seed_paper_id,
        status="succeeded",
        openreview_id=note.openreview_id or None,
        title=note.title,
        abstract=note.abstract,
        authors=list(note.authors),
        pdf=note.pdf,
        venueid=note.venueid,
        forum=note.forum,
        number=note.number,
        raw=note.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
