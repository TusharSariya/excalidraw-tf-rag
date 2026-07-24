"""Typed projection of a dblp record.

dblp's record API returns ``application/xml`` — a ``<dblp>`` wrapper around one
record element whose tag names the record type (``<inproceedings>`` for a
conference paper, ``<article>`` for a journal / CoRR paper, and other types dblp
uses: ``<incollection>``, ``<proceedings>``, ``<phdthesis>``, …). There is no
JSON variant (the ``.json`` URL 404s), so this module is the schema dblp's XML
declines to give: a frozen dataclass over the fields harvest reads, parsed from
the XML with tolerant, guarded access.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord``:

* **Tolerant, not strict.** Only the fields we know are projected; anything else
  in the XML is ignored — but nothing is lost, because the whole XML response is
  retained verbatim on ``raw`` (``{"xml": <text>}``) for the ``source_dblp.raw``
  jsonb column.
* **Everything optional.** dblp omits fields freely, and the field *element name*
  differs by record type — venue is ``<booktitle>`` on a conference record but
  ``<journal>`` on an article — so parsing checks both and every field defaults.

Key XML shape (verified against two live records — ``conf/nips/KrizhevskySH12``
(AlexNet, ``<inproceedings>``) and ``journals/corr/VaswaniSPUJGKP17`` (Attention,
``<article>``)):

* the wrapping record element's ``@key`` equals the requested ``dblp_id``;
* ``<title>`` text carries a trailing period, stripped here;
* ``<author>`` elements are repeated and ordered, each optionally carrying an
  ``@orcid`` attribute;
* there is NO ``<doi>`` element — a DOI, when present, appears only as an ``<ee>``
  (electronic-edition) URL containing ``doi.org``, and is frequently absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from xml.etree.ElementTree import Element, ParseError, fromstring

# Record types dblp wraps a record in. Used to locate the record element inside
# the <dblp> envelope; the parser falls back to the first element child so a new
# dblp type still parses rather than silently yielding nothing.
_RECORD_TAGS = frozenset(
    {
        "inproceedings",
        "article",
        "incollection",
        "proceedings",
        "book",
        "phdthesis",
        "mastersthesis",
        "www",
        "data",
    }
)


def _element_text(el: Element | None) -> str | None:
    """Full text of an element (including nested markup like ``<sub>``), stripped."""
    if el is None:
        return None
    text = "".join(el.itertext()).strip()
    return text or None


@dataclass(frozen=True)
class DblpAuthor:
    """One ``<author>`` — an ordered name, with the optional ORCID dblp carries."""

    name: str
    orcid: str | None = None

    @classmethod
    def from_element(cls, el: Element | None) -> DblpAuthor | None:
        if el is None:
            return None
        name = _element_text(el)
        if not name:
            return None
        return cls(name=name, orcid=el.get("orcid"))


@dataclass(frozen=True)
class DblpRecord:
    """A typed dblp record — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_xml`` takes the ``dblp_id`` that was requested and the raw XML response
    text. The whole XML is kept on ``raw`` (as ``{"xml": <text>}``) so
    ``source_dblp.raw`` stays a faithful copy even as this projection stays small.
    Returns ``None`` when the response carries no parseable record element.
    """

    dblp_id: str
    record_type: str | None = None
    title: str | None = None
    authors: list[DblpAuthor] = field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    ee: list[str] = field(default_factory=list)
    doi: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def author_names(self) -> list[str]:
        """Ordered author names, dropping the ORCID attribute."""
        return [a.name for a in self.authors]

    @classmethod
    def from_xml(cls, dblp_id: str, xml_text: str) -> DblpRecord | None:
        try:
            root = fromstring(xml_text)
        except ParseError:
            return None

        # The record element is the child of <dblp>; its tag is the record type.
        # Fall back to the first element child so an unknown dblp type still parses.
        record: Element | None = None
        for child in root:
            if child.tag in _RECORD_TAGS:
                record = child
                break
        if record is None:
            record = next(iter(root), None)
        if record is None:
            return None

        # venue element name differs by record type: <booktitle> (conf) vs <journal>.
        venue = _element_text(record.find("booktitle")) or _element_text(record.find("journal"))

        year_text = _element_text(record.find("year"))
        year: int | None = None
        if year_text:
            try:
                year = int(year_text)
            except ValueError:
                year = None

        ee = [text for e in record.findall("ee") if (text := _element_text(e))]

        # No <doi> element exists; a DOI, when present, is an <ee> URL on doi.org.
        doi: str | None = None
        for url in ee:
            if "doi.org/" in url:
                doi = url.split("doi.org/", 1)[1]
                break

        title = _element_text(record.find("title"))
        if title:
            title = title.rstrip(".").strip() or None

        return cls(
            dblp_id=record.get("key") or dblp_id,
            record_type=record.tag,
            title=title,
            authors=[a for a in map(DblpAuthor.from_element, record.findall("author")) if a],
            venue=venue,
            year=year,
            ee=ee,
            doi=doi,
            raw={"xml": xml_text},
        )
