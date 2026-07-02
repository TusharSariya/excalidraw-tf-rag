"""Pure matching helpers for citation-graph keying, shared by GLR/RLR `cite_enrich.py`.

DOI-only keying reaches only ~23% of the GLR corpus (docs/research-tool-decisions-log-
2026-07.md #27): many manifest items have no DOI, and many DOI lookups against OpenAlex
simply 404 (older graph-drawing PDFs, arXiv preprints registered under a different DOI
scheme, etc.). This module adds the two conservative fallback matchers each tool's
`harvest/cite_enrich.py` wires up against its own throttled OpenAlex client:

1. arXiv-ID extraction (``extract_arxiv_id``) — pulls a bare arXiv id out of whatever
   the manifest item carries (an explicit ``externalIds["ArXiv"]``, an arXiv-style DOI
   like ``10.48550/arxiv.2310.11511``, or an ``arxiv.org/abs|pdf/<id>`` URL) so the
   caller can look the work up via OpenAlex's ``works/arxiv:<id>`` endpoint.
2. Normalized-title matching (``best_title_match``) — given OpenAlex title-search
   candidates, accepts a match only when exactly one candidate clears a similarity
   threshold AND is within a year tolerance of the manifest item. Ambiguous (multiple
   qualifying candidates) or weak matches are rejected outright: false negatives (stay
   unkeyed, retry later) are preferred over false positives (a wrong citation edge is
   silently wrong forever).

No network code and no tool-specific imports here by design — both tools' `cite_enrich.py`
call these against their own live OpenAlex responses.
"""

from __future__ import annotations

import difflib
import re

# New-style arXiv ids: "2310.11511" (with optional "vN" version suffix, stripped).
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
# Old-style arXiv ids: "cs.DS/0512345" / "math/0512345" (with optional "vN" suffix).
_ARXIV_OLD_ID_RE = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)


def _search_arxiv_id(text: str) -> str | None:
    m = _ARXIV_ID_RE.search(text) or _ARXIV_OLD_ID_RE.search(text)
    return m.group(1) if m else None


def extract_arxiv_id(
    *,
    doi: str | None = None,
    external_ids: dict[str, str] | None = None,
    url: str | None = None,
) -> str | None:
    """Best-effort arXiv id extraction from whatever the manifest item carries.

    Checks, in order: an explicit ``externalIds["ArXiv"]`` (case-insensitive key), an
    arXiv-style DOI (``10.48550/arxiv.<id>``), and an ``arxiv.org`` URL. Returns the
    bare id with no version suffix (e.g. ``"2310.11511"``), or ``None``.
    """
    if external_ids:
        for key, value in external_ids.items():
            if key.lower() == "arxiv" and value:
                found = _search_arxiv_id(value)
                if found:
                    return found
    if doi and "arxiv" in doi.lower():
        found = _search_arxiv_id(doi.lower())
        if found:
            return found
    if url and "arxiv.org" in url.lower():
        found = _search_arxiv_id(url)
        if found:
            return found
    return None


def normalize_title(title: str) -> str:
    """Lowercase, strip all punctuation/whitespace for a conservative title-match key."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def title_similarity(a: str, b: str) -> float:
    """Similarity ratio in [0, 1] between two titles, computed over normalized forms."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def best_title_match(
    candidates: list[dict],
    *,
    target_title: str,
    target_year: int | None,
    title_key: str = "title",
    year_key: str = "year",
    min_similarity: float = 0.92,
    year_tolerance: int = 1,
) -> dict | None:
    """Pick the single unambiguous title match among ``candidates``, or ``None``.

    A candidate qualifies when its normalized-title similarity to ``target_title`` is
    ``>= min_similarity`` AND, whenever both years are known, the two are within
    ``year_tolerance``. Returns the qualifying candidate only when *exactly one*
    candidate qualifies; zero or multiple qualifying candidates both return ``None``
    (no match / ambiguous — conservative by design).
    """
    if not target_title:
        return None
    qualifying = []
    for cand in candidates:
        cand_title = cand.get(title_key)
        if not cand_title:
            continue
        if title_similarity(target_title, cand_title) < min_similarity:
            continue
        cand_year = cand.get(year_key)
        if target_year is not None and cand_year is not None:
            try:
                if abs(int(cand_year) - int(target_year)) > year_tolerance:
                    continue
            except (TypeError, ValueError):
                pass
        qualifying.append(cand)
    if len(qualifying) == 1:
        return qualifying[0]
    return None
