"""Corpus gap detection, shared by graph-layout-rag and rag-literature-rag.

Turns the citation graph built by `cite enrich` into a harvest-priority queue: works
that are cited BY in-corpus documents but are not themselves in the corpus, ranked by
how many *distinct* in-corpus documents cite them. A work with a high count here is a
canonical reference the corpus keeps pointing at without actually containing.

Same core-schema assumption as `citation_coverage.py`: `papers(oa_id, doi, doc_id,
title, year, cited_by_count, in_corpus, ...)` / `cites(src_oa, dst_oa,
is_influential)`, plus `paper_aliases(provider, external_id, oa_id)` for the arXiv id
fallback. Pure stdlib `sqlite3` — no tool-specific imports — so it lives here and both
tools' `cli.py` wire it into their `cite gaps` subcommand.

"In corpus" reuses the same `papers.in_corpus` flag `cite enrich` sets when a row is
resolved from (or matched back to) a manifest item — the same notion `cite coverage`
audits via `doc_id IS NOT NULL` (the two are kept in lockstep by `cite enrich`).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class GapWork:
    """An external work cited by the corpus but not (yet) harvested into it."""

    oa_id: str
    doi: str | None
    arxiv_id: str | None
    title: str | None
    year: int | None
    cited_by_count: int  # distinct in-corpus docs citing this work

    def to_dict(self) -> dict:
        return {
            "oa_id": self.oa_id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "year": self.year,
            "cited_by_count": self.cited_by_count,
        }

    def display_label(self) -> str:
        """Best-available human label: title, else DOI, else arXiv id, else OpenAlex id.

        Degrades gracefully — a gap work an OpenAlex `referenced_works` edge pointed at
        but that was never itself fetched/enriched may have no title at all.
        """
        if self.title:
            return self.title
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        return self.oa_id


def compute_gaps(db: sqlite3.Connection, *, min_citations: int = 2, top: int = 50) -> list[GapWork]:
    """Rank external (not-in-corpus) works by distinct in-corpus citer count.

    ``min_citations`` filters out one-off references (noise); ``top`` caps the result
    to the N highest-priority harvest candidates.
    """
    rows = db.execute(
        """
        SELECT
          c.dst_oa AS oa_id,
          pd.doi AS doi,
          pd.title AS title,
          pd.year AS year,
          (
            SELECT pa.external_id FROM paper_aliases pa
            WHERE pa.oa_id = c.dst_oa AND pa.provider = 'arxiv'
            LIMIT 1
          ) AS arxiv_id,
          COUNT(DISTINCT c.src_oa) AS in_corpus_citers
        FROM cites c
        JOIN papers ps ON c.src_oa = ps.oa_id AND ps.in_corpus = 1
        LEFT JOIN papers pd ON c.dst_oa = pd.oa_id
        WHERE pd.oa_id IS NULL OR pd.in_corpus = 0
        GROUP BY c.dst_oa
        HAVING COUNT(DISTINCT c.src_oa) >= :min_citations
        ORDER BY in_corpus_citers DESC, oa_id ASC
        LIMIT :top
        """,
        {"min_citations": min_citations, "top": top},
    ).fetchall()

    return [
        GapWork(
            oa_id=row["oa_id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            year=row["year"],
            cited_by_count=row["in_corpus_citers"],
        )
        for row in rows
    ]


# --- title resolution (--resolve-titles) -----------------------------------------
#
# `compute_gaps` only ever reports metadata already sitting in the citation store —
# a `dst_oa` that was never itself fetched/enriched has no title, DOI, or year, just
# a bare OpenAlex id. `resolve_gap_titles` batch-fills that metadata from the
# OpenAlex works endpoint (keyless polite pool, `mailto`-identified) and writes it
# back into `papers` (and `paper_aliases` for the arXiv id) so repeat runs are
# offline. Pure stdlib `urllib` — matches `rag_common.local_llm` / `remote_embed`,
# and rag-common has no HTTP client dependency of its own.

OPENALEX_WORKS_API = "https://api.openalex.org/works"
_RESOLVE_BATCH_SIZE = 50  # OpenAlex `filter=openalex_id:A|B|...` cap
_REQUEST_INTERVAL_S = 1.0  # modest politeness for the keyless pool
_RETRY_BACKOFF_S = 5.0

_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^v/]+)", re.IGNORECASE)


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_oa_id(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().rstrip("/")
    tail = s.rsplit("/", 1)[-1]
    return tail if tail and tail[0] in "Ww" else None


def _normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    low = s.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if low.startswith(prefix):
            s = s[len(prefix) :]
            break
    s = s.strip().lower().rstrip(").,;")
    return s if s.startswith("10.") else None


def _extract_arxiv_id(work: dict, doi: str | None) -> str | None:
    """Best-effort arXiv id: OpenAlex has no first-class `ids.arxiv` field, so this
    checks the `10.48550/arXiv.*` DOI form, then falls back to arxiv.org locations."""
    if doi:
        m = _ARXIV_DOI_RE.match(doi)
        if m:
            return m.group(1)
    locations = list(work.get("locations") or [])
    primary = work.get("primary_location")
    if primary:
        locations.append(primary)
    for loc in locations:
        url = (loc or {}).get("landing_page_url") or ""
        m = _ARXIV_URL_RE.search(url)
        if m:
            return m.group(1)
    return None


def _parse_work(work: dict) -> dict:
    oa_id = _normalize_oa_id(work.get("id"))
    doi = _normalize_doi((work.get("ids") or {}).get("doi") or work.get("doi"))
    return {
        "oa_id": oa_id,
        "title": work.get("display_name") or work.get("title"),
        "year": work.get("publication_year"),
        "doi": doi,
        "arxiv_id": _extract_arxiv_id(work, doi),
    }


def _fetch_singleton(oa_id: str, *, mailto: str) -> dict | None:
    """Fallback for a batch that failed: one work via `/works/{id}` (no `filter=`,
    so it's unaffected by the metered-search budget the list endpoint can hit)."""
    url = f"{OPENALEX_WORKS_API}/{oa_id}?{urllib.parse.urlencode({'mailto': mailto})}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"mailto:{mailto}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            work = json.loads(resp.read())
        return _parse_work(work)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None


def fetch_openalex_titles(
    oa_ids: list[str],
    *,
    mailto: str,
    batch_size: int = _RESOLVE_BATCH_SIZE,
    sleep_s: float = _REQUEST_INTERVAL_S,
) -> dict[str, dict]:
    """Batch-resolve OpenAlex work ids to {title, year, doi, arxiv_id}.

    Keyless polite-pool requests (``mailto`` query param), ≤``batch_size`` ids per
    call. Retries a 429 once with backoff; a batch that still fails (429 twice, or
    any other HTTP/network error) falls back to one singleton `/works/{id}` lookup
    per id in that batch (still keyless, still ``mailto``-identified) before giving
    up -- OpenAlex's `filter=`/search surface is metered separately from its plain
    id-lookup path, so a search-budget exhaustion doesn't necessarily block those.
    Any id that still can't be resolved is simply absent from the result -- callers
    keep the bare id for it, this function never raises.
    """
    resolved: dict[str, dict] = {}
    chunks = _chunked(list(oa_ids), batch_size)
    for i, chunk in enumerate(chunks):
        filter_val = "openalex_id:" + "|".join(chunk)
        params = {"filter": filter_val, "per-page": str(len(chunk)), "mailto": mailto}
        url = f"{OPENALEX_WORKS_API}?{urllib.parse.urlencode(params)}"
        batch_ok = False
        for attempt in range(2):  # initial try + one retry-on-429
            try:
                req = urllib.request.Request(url, headers={"User-Agent": f"mailto:{mailto}"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                for work in data.get("results", []):
                    parsed = _parse_work(work)
                    if parsed["oa_id"]:
                        resolved[parsed["oa_id"]] = parsed
                batch_ok = True
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt == 0:
                    time.sleep(_RETRY_BACKOFF_S)
                    continue
                print(
                    f"cite gaps --resolve-titles: OpenAlex batch {i + 1}/{len(chunks)} "
                    f"failed ({exc.code}) -- falling back to singleton lookups.",
                    file=sys.stderr,
                )
                break
            except urllib.error.URLError as exc:
                print(
                    f"cite gaps --resolve-titles: OpenAlex batch {i + 1}/{len(chunks)} "
                    f"unreachable ({exc}) -- falling back to singleton lookups.",
                    file=sys.stderr,
                )
                break
        if not batch_ok:
            for oa_id in chunk:
                parsed = _fetch_singleton(oa_id, mailto=mailto)
                if parsed and parsed["oa_id"]:
                    resolved[parsed["oa_id"]] = parsed
                time.sleep(sleep_s)
        if i < len(chunks) - 1:
            time.sleep(sleep_s)
    return resolved


def _write_back(db: sqlite3.Connection, resolved: dict[str, dict]) -> None:
    """Persist resolved metadata into `papers` (+ `paper_aliases` for arXiv ids), using
    the same COALESCE-on-conflict shape as each tool's `citation_store.upsert_paper` /
    `upsert_alias` so a resolve run never clobbers metadata a fuller upsert already set."""
    for oa_id, meta in resolved.items():
        db.execute(
            """
            INSERT INTO papers(oa_id, doi, title, year, in_corpus)
            VALUES(:oa, :doi, :title, :year, 0)
            ON CONFLICT(oa_id) DO UPDATE SET
              doi=COALESCE(excluded.doi, papers.doi),
              title=COALESCE(excluded.title, papers.title),
              year=COALESCE(excluded.year, papers.year)
            """,
            {
                "oa": oa_id,
                "doi": meta.get("doi"),
                "title": meta.get("title"),
                "year": meta.get("year"),
            },
        )
        if meta.get("arxiv_id"):
            db.execute(
                """
                INSERT INTO paper_aliases(provider, external_id, oa_id) VALUES('arxiv', :ext, :oa)
                ON CONFLICT(provider, external_id) DO UPDATE SET oa_id=excluded.oa_id
                """,
                {"ext": meta["arxiv_id"], "oa": oa_id},
            )
    db.commit()


def resolve_gap_titles(
    db: sqlite3.Connection,
    gaps: list[GapWork],
    *,
    mailto: str,
    fetch_fn=fetch_openalex_titles,
    batch_size: int = _RESOLVE_BATCH_SIZE,
    sleep_s: float = _REQUEST_INTERVAL_S,
) -> list[GapWork]:
    """Fill in missing titles/years/DOIs/arXiv ids for `gaps` via OpenAlex, persisting
    the result into the citation store (`_write_back`) so a repeat run is offline.

    Only gap works still missing a title are fetched -- a gap already resolved (by an
    earlier `--resolve-titles` run, or because `cite enrich` happened to touch it) is
    left untouched and costs no network call.
    """
    needs_fetch = [g.oa_id for g in gaps if not g.title]
    if not needs_fetch:
        return gaps

    resolved = fetch_fn(needs_fetch, mailto=mailto, batch_size=batch_size, sleep_s=sleep_s)
    if not resolved:
        return gaps

    _write_back(db, resolved)

    updated: list[GapWork] = []
    for g in gaps:
        meta = resolved.get(g.oa_id)
        if not meta:
            updated.append(g)
            continue
        updated.append(
            GapWork(
                oa_id=g.oa_id,
                doi=meta.get("doi") or g.doi,
                arxiv_id=meta.get("arxiv_id") or g.arxiv_id,
                title=meta.get("title") or g.title,
                year=meta.get("year") or g.year,
                cited_by_count=g.cited_by_count,
            )
        )
    return updated
