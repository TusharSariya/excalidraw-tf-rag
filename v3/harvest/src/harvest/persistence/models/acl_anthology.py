"""Table-shaped DTO for public.source_acl_anthology + AclAnthologyRecord -> Write mapper.

Mirrors ``persistence.models.unpaywall`` / ``…openalex``: one row group, keyed by the
seed's bigint ``paper_id``. The typed columns are the queryable subset; the whole
parsed BibTeX entry (plus the real ``raw_bibtex`` bytes) is kept on ``raw``.

``AclAnthologyRecord`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.acl_anthology import AclAnthologyRecord


class SourceAclAnthologyRow(TableRow):
    """Insert/upsert shape for public.source_acl_anthology (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``authors`` is a text[] of ``Last, First``
    names; ``raw`` is jsonb holding the whole parsed entry. The scalar columns are
    the promoted, queryable ACL Anthology fields.
    """

    __table__ = "source_acl_anthology"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    anthology_id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    bibkey: str | None = None
    entry_type: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def acl_anthology_write(
    seed_paper_id: int,
    rec: AclAnthologyRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_acl_anthology`` success ``Write`` for one fetched record.

    A single row group keyed by the seed's ``paper_id`` — ACL Anthology has no
    citation graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceAclAnthologyRow(
        paper_id=seed_paper_id,
        status="succeeded",
        anthology_id=rec.anthology_id or None,
        title=rec.title,
        authors=rec.authors or None,
        year=rec.year,
        venue=rec.venue,
        doi=rec.doi,
        url=rec.url,
        pdf_url=rec.pdf_url,
        bibkey=rec.bibkey,
        entry_type=rec.entry_type,
        raw=rec.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
