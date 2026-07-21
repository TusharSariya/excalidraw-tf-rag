"""Typed projection of an OpenAlex Work.

pyalex returns a Work as a bare ``dict`` subclass — no schema, no validation, a typo
in a key silently yields ``None``. This module is the schema pyalex declines to give:
frozen dataclasses over the fields we actually read, each with a tolerant ``from_dict``.

Design choices, mirroring ``ArxivRecord`` / ``S2Record``:

* **Tolerant, not strict.** Unlike ``S2Record`` (which rejects unknown top-level keys),
  this keeps only the fields it knows and ignores the rest — OpenAlex adds fields
  often, and a harvester should not break when it does. Nothing is lost regardless:
  the whole response is retained on ``raw`` for the ``source_openalex.raw`` jsonb column.
* **Everything optional.** OpenAlex omits/nulls fields freely (``best_oa_location`` is
  often ``None``, ``biblio.issue`` empty, …), so every field defaults and nested access
  is guarded.

Field names/nesting verified against a live Work response (pyalex 0.21 / OpenAlex API).
See https://docs.openalex.org/api-entities/works/work-object for the full catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkIds:
    """The ``ids`` block — the work's identity across namespaces."""

    openalex: str | None = None
    doi: str | None = None
    mag: str | None = None
    pmid: str | None = None
    pmcid: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WorkIds | None:
        if not data:
            return None
        return cls(
            openalex=data.get("openalex"),
            doi=data.get("doi"),
            mag=data.get("mag"),
            pmid=data.get("pmid"),
            pmcid=data.get("pmcid"),
        )


@dataclass(frozen=True)
class DehydratedSource:
    """A location's ``source`` — the journal / repository hosting the copy."""

    id: str | None = None
    display_name: str | None = None
    issn_l: str | None = None
    issn: list[str] | None = None
    type: str | None = None
    host_organization: str | None = None
    host_organization_name: str | None = None
    is_oa: bool | None = None
    is_in_doaj: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DehydratedSource | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            display_name=data.get("display_name"),
            issn_l=data.get("issn_l"),
            issn=data.get("issn"),
            type=data.get("type"),
            host_organization=data.get("host_organization"),
            host_organization_name=data.get("host_organization_name"),
            is_oa=data.get("is_oa"),
            is_in_doaj=data.get("is_in_doaj"),
        )


@dataclass(frozen=True)
class Location:
    """One hosted copy — ``primary_location``, ``best_oa_location``, or a ``locations`` item."""

    is_oa: bool | None = None
    landing_page_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    version: str | None = None
    is_published: bool | None = None
    is_accepted: bool | None = None
    source: DehydratedSource | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Location | None:
        if not data:
            return None
        return cls(
            is_oa=data.get("is_oa"),
            landing_page_url=data.get("landing_page_url"),
            pdf_url=data.get("pdf_url"),
            license=data.get("license"),
            version=data.get("version"),
            is_published=data.get("is_published"),
            is_accepted=data.get("is_accepted"),
            source=DehydratedSource.from_dict(data.get("source")),
        )


@dataclass(frozen=True)
class OpenAccess:
    is_oa: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None
    any_repository_has_fulltext: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OpenAccess | None:
        if not data:
            return None
        return cls(
            is_oa=data.get("is_oa"),
            oa_status=data.get("oa_status"),
            oa_url=data.get("oa_url"),
            any_repository_has_fulltext=data.get("any_repository_has_fulltext"),
        )


@dataclass(frozen=True)
class AuthorRef:
    id: str | None = None
    display_name: str | None = None
    orcid: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AuthorRef | None:
        if not data:
            return None
        return cls(
            id=data.get("id"), display_name=data.get("display_name"), orcid=data.get("orcid")
        )


@dataclass(frozen=True)
class Institution:
    id: str | None = None
    display_name: str | None = None
    ror: str | None = None
    country_code: str | None = None
    type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Institution | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            display_name=data.get("display_name"),
            ror=data.get("ror"),
            country_code=data.get("country_code"),
            type=data.get("type"),
        )


@dataclass(frozen=True)
class Authorship:
    author_position: str | None = None
    author: AuthorRef | None = None
    institutions: list[Institution] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    is_corresponding: bool | None = None
    raw_author_name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Authorship | None:
        if not data:
            return None
        return cls(
            author_position=data.get("author_position"),
            author=AuthorRef.from_dict(data.get("author")),
            institutions=[
                i for i in map(Institution.from_dict, data.get("institutions") or []) if i
            ],
            countries=list(data.get("countries") or []),
            is_corresponding=data.get("is_corresponding"),
            raw_author_name=data.get("raw_author_name"),
        )


@dataclass(frozen=True)
class Biblio:
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Biblio | None:
        if not data:
            return None
        return cls(
            volume=data.get("volume"),
            issue=data.get("issue"),
            first_page=data.get("first_page"),
            last_page=data.get("last_page"),
        )


@dataclass(frozen=True)
class TopicLevel:
    """A rung of the topic hierarchy: ``domain`` / ``field`` / ``subfield``."""

    id: str | None = None
    display_name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TopicLevel | None:
        if not data:
            return None
        return cls(id=data.get("id"), display_name=data.get("display_name"))


@dataclass(frozen=True)
class Topic:
    id: str | None = None
    display_name: str | None = None
    score: float | None = None
    domain: TopicLevel | None = None
    field: TopicLevel | None = None
    subfield: TopicLevel | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Topic | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            display_name=data.get("display_name"),
            score=data.get("score"),
            domain=TopicLevel.from_dict(data.get("domain")),
            field=TopicLevel.from_dict(data.get("field")),
            subfield=TopicLevel.from_dict(data.get("subfield")),
        )


@dataclass(frozen=True)
class CountByYear:
    year: int | None = None
    cited_by_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CountByYear | None:
        if not data:
            return None
        return cls(year=data.get("year"), cited_by_count=data.get("cited_by_count"))


@dataclass(frozen=True)
class Keyword:
    id: str | None = None
    display_name: str | None = None
    score: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Keyword | None:
        if not data:
            return None
        return cls(
            id=data.get("id"), display_name=data.get("display_name"), score=data.get("score")
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class OpenAlexWork:
    """A typed OpenAlex Work — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes the plain dict pyalex hands back (``Works()[id]``). Keep the
    whole dict on ``raw`` so ``source_openalex.raw`` stays a faithful copy even as this
    projection stays small.
    """

    openalex_id: str
    title: str
    doi: str | None = None
    display_name: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    language: str | None = None
    type: str | None = None
    ids: WorkIds | None = None
    primary_location: Location | None = None
    best_oa_location: Location | None = None
    locations: list[Location] = field(default_factory=list)
    locations_count: int | None = None
    open_access: OpenAccess | None = None
    authorships: list[Authorship] = field(default_factory=list)
    corresponding_author_ids: list[str] = field(default_factory=list)
    cited_by_count: int | None = None
    counts_by_year: list[CountByYear] = field(default_factory=list)
    fwci: float | None = None
    biblio: Biblio | None = None
    is_retracted: bool | None = None
    is_paratext: bool | None = None
    primary_topic: Topic | None = None
    topics: list[Topic] = field(default_factory=list)
    keywords: list[Keyword] = field(default_factory=list)
    referenced_works: list[str] = field(default_factory=list)
    referenced_works_count: int | None = None
    related_works: list[str] = field(default_factory=list)
    has_fulltext: bool | None = None
    abstract_inverted_index: dict[str, list[int]] | None = None
    created_date: str | None = None
    updated_date: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def best_oa_pdf_url(self) -> str | None:
        """The open-access PDF URL, if OpenAlex found one."""
        return self.best_oa_location.pdf_url if self.best_oa_location else None

    @property
    def abstract(self) -> str | None:
        """Reconstruct plain-text abstract from OpenAlex's inverted index.

        OpenAlex ships the abstract as ``{word: [positions]}`` (for copyright reasons);
        invert it back to running text. Returns None when no abstract is present.
        """
        inv = self.abstract_inverted_index
        if not inv:
            return None
        positions: dict[int, str] = {}
        for word, idxs in inv.items():
            for i in idxs:
                positions[i] = word
        return " ".join(positions[i] for i in sorted(positions))

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> OpenAlexWork:
        return cls(
            openalex_id=result.get("id") or "",
            title=result.get("title") or "",
            doi=result.get("doi"),
            display_name=result.get("display_name"),
            publication_year=result.get("publication_year"),
            publication_date=result.get("publication_date"),
            language=result.get("language"),
            type=result.get("type"),
            ids=WorkIds.from_dict(result.get("ids")),
            primary_location=Location.from_dict(result.get("primary_location")),
            best_oa_location=Location.from_dict(result.get("best_oa_location")),
            locations=_list(Location.from_dict, result.get("locations")),
            locations_count=result.get("locations_count"),
            open_access=OpenAccess.from_dict(result.get("open_access")),
            authorships=_list(Authorship.from_dict, result.get("authorships")),
            corresponding_author_ids=list(result.get("corresponding_author_ids") or []),
            cited_by_count=result.get("cited_by_count"),
            counts_by_year=_list(CountByYear.from_dict, result.get("counts_by_year")),
            fwci=result.get("fwci"),
            biblio=Biblio.from_dict(result.get("biblio")),
            is_retracted=result.get("is_retracted"),
            is_paratext=result.get("is_paratext"),
            primary_topic=Topic.from_dict(result.get("primary_topic")),
            topics=_list(Topic.from_dict, result.get("topics")),
            keywords=_list(Keyword.from_dict, result.get("keywords")),
            referenced_works=list(result.get("referenced_works") or []),
            referenced_works_count=result.get("referenced_works_count"),
            related_works=list(result.get("related_works") or []),
            has_fulltext=result.get("has_fulltext"),
            abstract_inverted_index=result.get("abstract_inverted_index"),
            created_date=result.get("created_date"),
            updated_date=result.get("updated_date"),
            # pyalex returns a Work (a dict subclass); store a plain dict so the `raw`
            # jsonb column serialises predictably.
            raw=dict(result),
        )
