"""Table-shaped DTO for public.source_internet_archive + InternetArchiveItem -> Write.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's bigint
``paper_id``. The typed columns are the queryable subset; the whole ``/metadata``
response is kept on ``raw``. IA has no citation graph, so there is nothing to fan out
(contrast ``s2_write``).

``InternetArchiveItem`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.internet_archive import InternetArchiveItem


class SourceInternetArchiveRow(TableRow):
    """Insert/upsert shape for public.source_internet_archive (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``creator`` / ``subject`` are text[]; ``raw`` is
    jsonb; the scalar columns are the promoted, queryable IA fields. ``pdf_url`` is the
    constructed download URL — the worker persists it but never fetches the binary.
    """

    __table__ = "source_internet_archive"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    ia_identifier: str | None = None
    title: str | None = None
    creator: list[str] = []
    date: str | None = None
    publicdate: str | None = None
    year: int | None = None
    description: str | None = None
    subject: list[str] = []
    licenseurl: str | None = None
    mediatype: str | None = None
    pdf_file_name: str | None = None
    pdf_url: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def internet_archive_write(
    seed_paper_id: int,
    item: InternetArchiveItem,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_internet_archive`` success ``Write`` for one fetched item.

    A single row group keyed by the seed's ``paper_id``. ``description`` (a str-or-list
    metadata field normalised to a list on the record) is collapsed to text here; the
    full listing stays on ``raw``.
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    pdf = item.pdf_file
    row = SourceInternetArchiveRow(
        paper_id=seed_paper_id,
        status="succeeded",
        ia_identifier=item.identifier or None,
        title=item.title,
        creator=item.creator,
        date=item.date,
        publicdate=item.publicdate,
        year=item.year,
        description="\n".join(item.description) if item.description else None,
        subject=item.subject,
        licenseurl=item.licenseurl,
        mediatype=item.mediatype,
        pdf_file_name=pdf.name if pdf else None,
        pdf_url=item.pdf_url,
        raw=item.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
