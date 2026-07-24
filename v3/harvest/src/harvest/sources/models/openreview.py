"""Typed projection of an OpenReview Note (api2 REST ``/notes``).

The api2 ``/notes`` endpoint returns notes whose ``content`` is a bag of
``{field: {"value": ...}}`` wrappers — no schema, and a wrong key silently reads
the wrapper (or ``None``) instead of the value. This module is that missing
schema: a frozen dataclass over the fields harvest reads, with a tolerant
``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord``:

* **Tolerant, not strict.** Keep only the fields we know; ignore the rest.
  Nothing is lost regardless — the whole note JSON is retained on ``raw`` for the
  ``source_openreview.raw`` jsonb column.
* **Everything optional.** OpenReview omits/nulls content fields freely (a
  desk-rejected submission may carry no ``pdf``/``venueid``), so every field
  defaults and every nested access is guarded.
* **API2 content wrapping.** In the api2 surface a content field is
  ``content[key]["value"]`` (see openreview-py docs: *"Always access content
  fields using ['value']"*). ``_content_value`` unwraps that while still
  tolerating a bare, unwrapped value from the older v1 shape.

Field names (``id, title, abstract, authors, pdf, venueid``) follow the source
plan; the live API is behind a challenge gate, so the exact response could not be
recorded — the projection is intentionally small and every field is optional so a
shape surprise degrades to ``None`` rather than an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _content_value(content: dict[str, Any] | None, key: str) -> Any:
    """Unwrap ``content[key]`` for the api2 ``{"value": ...}`` shape.

    Returns ``None`` when the field is absent. Tolerates a bare (v1-style) value
    that is not wrapped in a ``{"value": ...}`` dict.
    """
    if not content:
        return None
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _str_list(value: Any) -> list[str]:
    """Coerce an authors-like value into a list of strings, dropping blanks."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


@dataclass(frozen=True)
class OpenReviewNote:
    """A typed OpenReview Note — the subset harvest reads, plus ``raw`` for persist.

    ``from_dict`` takes one note object from the api2 ``/notes`` JSON response.
    ``openreview_id`` is the note id — for a submission it equals its ``forum``
    id, which is the identity seeds carry.
    """

    openreview_id: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    pdf: str | None = None
    venueid: str | None = None
    forum: str | None = None
    number: int | None = None
    invitations: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenReviewNote:
        """Build from a note JSON dict (``Note.to_json()``)."""
        content = data.get("content")
        # api2 carries `invitations` (list); v1 carried `invitation` (scalar).
        invitations = data.get("invitations")
        if invitations is None and data.get("invitation") is not None:
            invitations = [data["invitation"]]
        return cls(
            openreview_id=data.get("id") or "",
            title=_content_value(content, "title"),
            abstract=_content_value(content, "abstract"),
            authors=_str_list(_content_value(content, "authors")),
            pdf=_content_value(content, "pdf"),
            venueid=_content_value(content, "venueid"),
            forum=data.get("forum"),
            number=data.get("number"),
            invitations=_str_list(invitations),
            raw=dict(data),
        )
