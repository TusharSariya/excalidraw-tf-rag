"""Typed projection of an Internet Archive item's metadata response.

``internetarchive.get_item(identifier)`` fetches ``GET https://archive.org/metadata/{identifier}``
under the hood and exposes the parsed JSON on ``Item.item_metadata`` — a bare ``dict``
with no schema. This module is the schema the SDK declines to give: frozen dataclasses
over the fields we actually read, each with a tolerant ``from_dict`` / ``from_result``.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest — IA adds
  and omits metadata keys freely. Nothing is lost regardless: the whole response is
  retained on ``raw`` for the ``source_internet_archive.raw`` jsonb column.
* **Everything optional.** IA scans routinely omit ``creator`` and ``date`` (this very
  sample does), so every metadata field defaults and nested access is guarded.
* **str-or-list fields are normalised to lists.** IA returns ``creator`` / ``subject`` /
  ``description`` as a bare string when there is one value and a list when there are
  several. ``_as_str_list`` collapses both to ``list[str]`` so callers see one shape.

Field names / nesting verified against a live metadata response (saved to
``tests/fixtures/internet_archive_sample.json``). The response has the item metadata
under the top-level ``metadata`` block, with ``files`` / ``server`` / ``dir`` at the top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Canonical per-item download host. The PDF url is built from this + identifier + the
# chosen PDF filename; it is NOT a field in the response (see ``pdf_url``).
_DOWNLOAD_BASE = "https://archive.org/download"

# First 4-digit run that looks like a year, used to derive a year from ``date`` /
# ``publicdate`` ('2025-10-08 17:57:36' -> 2025).
_YEAR_RE = re.compile(r"(\d{4})")


def _as_str_list(value: Any) -> list[str]:
    """Normalise a str-or-list-or-None metadata value to ``list[str]``.

    IA returns single-valued fields as a bare string and multi-valued ones as a list;
    this collapses both (and None) to a list, dropping empties.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v) != ""]
    return [str(value)]


def _year_from(*candidates: str | None) -> int | None:
    """First 4-digit year found across the candidate date strings, in order."""
    for candidate in candidates:
        if not candidate:
            continue
        m = _YEAR_RE.search(candidate)
        if m:
            return int(m.group(1))
    return None


@dataclass(frozen=True)
class IAFile:
    """One entry of the item's ``files`` listing."""

    name: str | None = None
    format: str | None = None
    source: str | None = None
    size: str | None = None
    md5: str | None = None
    sha1: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> IAFile | None:
        if not data:
            return None
        return cls(
            name=data.get("name"),
            format=data.get("format"),
            source=data.get("source"),
            size=data.get("size"),
            md5=data.get("md5"),
            sha1=data.get("sha1"),
        )

    @property
    def is_pdf(self) -> bool:
        """True when this file's format names a PDF ('Text PDF', not 'application/pdf')."""
        return bool(self.format) and "pdf" in self.format.lower()


@dataclass(frozen=True)
class InternetArchiveItem:
    """A typed IA item — the subset harvest reads, plus ``raw`` for lossless persist.

    ``from_result`` takes the full ``/metadata`` response dict (``Item.item_metadata``).
    The whole dict is kept on ``raw`` so ``source_internet_archive.raw`` stays faithful
    even as this projection stays small.
    """

    identifier: str
    title: str | None = None
    creator: list[str] = field(default_factory=list)
    date: str | None = None
    publicdate: str | None = None
    description: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    collection: list[str] = field(default_factory=list)
    licenseurl: str | None = None
    mediatype: str | None = None
    server: str | None = None
    dir: str | None = None
    files: list[IAFile] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def pdf_file(self) -> IAFile | None:
        """The item's PDF file, preferring the original scan over derivatives.

        Iterate ``files``, keep the ones whose format matches PDF, and prefer
        ``source == 'original'`` (the uploaded PDF) over a derived one. Returns None
        when the item carries no PDF.
        """
        pdfs = [f for f in self.files if f.is_pdf and f.name]
        if not pdfs:
            return None
        for f in pdfs:
            if f.source == "original":
                return f
        return pdfs[0]

    @property
    def pdf_url(self) -> str | None:
        """Canonical download URL for the chosen PDF, or None if there is no PDF.

        Constructed as ``{_DOWNLOAD_BASE}/{identifier}/{name}`` — NOT a response field.
        The worker does metadata only and never fetches the binary.
        """
        pdf = self.pdf_file
        if pdf is None or not pdf.name:
            return None
        return f"{_DOWNLOAD_BASE}/{self.identifier}/{pdf.name}"

    @property
    def year(self) -> int | None:
        """Publication year from ``date``, falling back to the ``publicdate`` upload ts."""
        return _year_from(self.date, self.publicdate)

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> InternetArchiveItem:
        """Build from the full ``/metadata`` response dict (``Item.item_metadata``)."""
        meta = result.get("metadata") or {}
        return cls(
            identifier=meta.get("identifier") or "",
            title=meta.get("title"),
            creator=_as_str_list(meta.get("creator")),
            date=meta.get("date"),
            publicdate=meta.get("publicdate"),
            description=_as_str_list(meta.get("description")),
            subject=_as_str_list(meta.get("subject")),
            collection=_as_str_list(meta.get("collection")),
            licenseurl=meta.get("licenseurl"),
            mediatype=meta.get("mediatype"),
            server=result.get("server"),
            dir=result.get("dir"),
            files=[f for f in map(IAFile.from_dict, result.get("files") or []) if f is not None],
            # Keep the whole response so the `raw` jsonb column is a faithful copy.
            raw=dict(result),
        )
