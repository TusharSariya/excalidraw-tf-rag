"""Typed projection of a CORE v3 ``work``.

The CORE v3 API (https://api.core.ac.uk/docs/v3) returns a work as a bare JSON
object — no schema, no validation. This module is the schema CORE declines to
give: frozen dataclasses over the fields harvest actually reads, each with a
tolerant ``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  CORE adds/renames fields between releases, and a harvester should not break
  when it does. Nothing is lost regardless: the whole response is retained on
  ``raw`` for the ``source_core.raw`` jsonb column.
* **Everything optional.** CORE omits/nulls fields freely (``downloadUrl`` is
  often ``None`` for metadata-only works, ``language`` empty, …), so every field
  defaults and nested access is guarded.

The probe that seeded this task could not reach CORE live (``CORE_API_KEY`` was
absent), so the field set below is typed against the documented CORE v3 ``work``
object rather than a captured sample. The queryable subset is projected to typed
attributes; the OA fulltext URL / download id (``downloadUrl`` /
``fullTextIdentifier`` / the numeric ``id``) is the point of the source. Re-run
the probe with a key to confirm/extend these fields against a real response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoreAuthor:
    """One entry of a work's ``authors`` array."""

    name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoreAuthor | None:
        if not data:
            return None
        return cls(name=data.get("name"))


@dataclass(frozen=True)
class CoreLanguage:
    """The work's ``language`` block."""

    code: str | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoreLanguage | None:
        if not data:
            return None
        return cls(code=data.get("code"), name=data.get("name"))


@dataclass(frozen=True)
class CoreDataProvider:
    """One entry of a work's ``dataProviders`` array — the repository hosting the copy."""

    id: int | None = None
    name: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoreDataProvider | None:
        if not data:
            return None
        return cls(id=data.get("id"), name=data.get("name"), url=data.get("url"))


@dataclass(frozen=True)
class CoreLink:
    """One entry of a work's ``links`` array."""

    type: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoreLink | None:
        if not data:
            return None
        return cls(type=data.get("type"), url=data.get("url"))


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class CoreWork:
    """A typed CORE v3 work — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes the plain dict CORE hands back (a ``search/works`` result
    item, or a ``works/{id}`` body). Keep the whole dict on ``raw`` so
    ``source_core.raw`` stays a faithful copy even as this projection stays small.
    """

    core_id: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    download_url: str | None = None
    full_text_identifier: str | None = None
    source_fulltext_urls: list[str] = field(default_factory=list)
    year_published: int | None = None
    published_date: str | None = None
    publisher: str | None = None
    document_type: str | None = None
    language: CoreLanguage | None = None
    authors: list[CoreAuthor] = field(default_factory=list)
    data_providers: list[CoreDataProvider] = field(default_factory=list)
    links: list[CoreLink] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def oa_fulltext_url(self) -> str | None:
        """Best open-access fulltext URL: the download URL, else the fulltext identifier."""
        return self.download_url or self.full_text_identifier

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> CoreWork:
        # CORE's numeric work id doubles as its download id; keep it as text so the
        # `raw` jsonb and the typed column agree regardless of int/str in the payload.
        raw_id = result.get("id")
        return cls(
            core_id=str(raw_id) if raw_id is not None else None,
            doi=result.get("doi"),
            title=result.get("title"),
            abstract=result.get("abstract"),
            download_url=result.get("downloadUrl"),
            full_text_identifier=result.get("fullTextIdentifier"),
            source_fulltext_urls=list(result.get("sourceFulltextUrls") or []),
            year_published=result.get("yearPublished"),
            published_date=result.get("publishedDate"),
            publisher=result.get("publisher"),
            document_type=result.get("documentType"),
            language=CoreLanguage.from_dict(result.get("language")),
            authors=_list(CoreAuthor.from_dict, result.get("authors")),
            data_providers=_list(CoreDataProvider.from_dict, result.get("dataProviders")),
            links=_list(CoreLink.from_dict, result.get("links")),
            raw=dict(result),
        )
