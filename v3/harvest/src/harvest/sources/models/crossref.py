"""Typed projection of a Crossref ``/works`` item.

Crossref returns each work as an untyped JSON object under ``message.items[]`` —
no schema, keys hyphenated and heterogeneous (``is-referenced-by-count``,
``container-title``, a ``reference[]`` list where each entry carries a different
subset of fields). This module is the schema Crossref does not give: frozen
dataclasses over the fields harvest actually reads, each with a tolerant
``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  Crossref items carry dozens of fields we do not query. Nothing is lost: the
  whole item is retained on ``raw`` for the ``source_crossref.raw`` jsonb column,
  and each reference entry keeps its own ``raw`` for ``crossref_reference.raw``.
* **Everything optional.** Crossref omits fields freely — ``title`` and
  ``container-title`` are arrays that may be empty, ``abstract`` is usually
  absent, ``ISSN`` is present only for journal articles, and a ``reference[]``
  entry carries a ``DOI`` only ~30-70% of the time — so every field defaults and
  nested access is guarded.

Field names/nesting verified against a live ``/works?filter=doi:...`` response.
DOIs echo back LOWERCASED regardless of the queried case; callers case-fold when
re-keying to the requested id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _first(seq: Any) -> Any | None:
    """Crossref ships ``title`` / ``container-title`` as arrays; take ``[0]``."""
    if isinstance(seq, list) and seq:
        return seq[0]
    return None


def _issued_year(data: dict[str, Any] | None) -> int | None:
    """Publication year from ``issued`` — shape ``{date-parts: [[YYYY, MM, DD?]]}``.

    Preferred over ``published-print`` / ``created`` for publication year.
    """
    if not data:
        return None
    parts = data.get("date-parts")
    if not isinstance(parts, list) or not parts:
        return None
    first = parts[0]
    if not isinstance(first, list) or not first:
        return None
    year = first[0]
    return year if isinstance(year, int) else None


@dataclass(frozen=True)
class CrossrefAuthor:
    """One entry of ``author[]`` — ``{given, family, sequence, affiliation, ORCID?}``."""

    given: str | None = None
    family: str | None = None
    sequence: str | None = None
    orcid: str | None = None
    affiliation: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str | None:
        """Build a display name from ``given`` + ``family``; None if neither present."""
        parts = [p for p in (self.given, self.family) if p]
        return " ".join(parts) if parts else None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CrossrefAuthor | None:
        if not data:
            return None
        affiliation = [
            a.get("name")
            for a in (data.get("affiliation") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        return cls(
            given=data.get("given"),
            family=data.get("family"),
            sequence=data.get("sequence"),
            orcid=data.get("ORCID"),
            affiliation=affiliation,
        )


@dataclass(frozen=True)
class CrossrefReference:
    """One entry of ``reference[]`` — an outbound reference, persisted as an edge row.

    Heterogeneous: every entry has ``key``; ``DOI`` is present only ~30-70% of the
    time (lowercased, like the top-level DOI). DOI-less entries are unstructured /
    partially structured (``article-title`` / ``journal-title`` / ``unstructured``).
    The whole entry is kept on ``raw`` for the ``crossref_reference.raw`` column.
    """

    key: str | None = None
    doi: str | None = None
    doi_asserted_by: str | None = None
    article_title: str | None = None
    journal_title: str | None = None
    author: str | None = None
    year: str | None = None
    volume: str | None = None
    first_page: str | None = None
    unstructured: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CrossrefReference | None:
        if not data:
            return None
        return cls(
            key=data.get("key"),
            doi=data.get("DOI"),
            doi_asserted_by=data.get("doi-asserted-by"),
            article_title=data.get("article-title"),
            journal_title=data.get("journal-title"),
            author=data.get("author"),
            year=data.get("year"),
            volume=data.get("volume"),
            first_page=data.get("first-page"),
            unstructured=data.get("unstructured"),
            raw=dict(data),
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class CrossrefWork:
    """A typed Crossref work — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes one ``message.items[]`` object. Keep the whole item on
    ``raw`` so ``source_crossref.raw`` stays a faithful copy even as this
    projection stays small. ``doi`` echoes back LOWERCASED from Crossref.
    """

    doi: str
    title: str | None = None
    container_title: str | None = None
    type: str | None = None
    publisher: str | None = None
    url: str | None = None
    issued_year: int | None = None
    is_referenced_by_count: int | None = None
    references_count: int | None = None
    page: str | None = None
    abstract: str | None = None
    issn: list[str] = field(default_factory=list)
    authors: list[CrossrefAuthor] = field(default_factory=list)
    references: list[CrossrefReference] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> CrossrefWork:
        # references-count is the current key; reference-count is the legacy alias.
        references_count = result.get("references-count")
        if references_count is None:
            references_count = result.get("reference-count")
        return cls(
            doi=result.get("DOI") or "",
            title=_first(result.get("title")),
            container_title=_first(result.get("container-title")),
            type=result.get("type"),
            publisher=result.get("publisher"),
            url=result.get("URL"),
            issued_year=_issued_year(result.get("issued")),
            is_referenced_by_count=result.get("is-referenced-by-count"),
            references_count=references_count,
            page=result.get("page"),
            abstract=result.get("abstract"),
            issn=list(result.get("ISSN") or []),
            authors=_list(CrossrefAuthor.from_dict, result.get("author")),
            references=_list(CrossrefReference.from_dict, result.get("reference")),
            raw=dict(result),
        )
