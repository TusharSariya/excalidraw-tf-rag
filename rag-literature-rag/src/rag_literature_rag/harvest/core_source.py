"""CORE open-access full-text harvest (api.core.ac.uk v3)."""

from __future__ import annotations

import os
import re

from rag_literature_rag.harvest.archive_fallback import CORE_SEARCH
from rag_literature_rag.harvest.download import download_to_file
from rag_literature_rag.harvest.log import get_logger
from rag_literature_rag.harvest.parallel import parallel_map
from rag_literature_rag.harvest.providers import CORE, OutcomeKind
from rag_literature_rag.harvest.relevance import is_layout_relevant
from rag_literature_rag.manifest import ManifestItem, relative_local_path, slug_id
from rag_literature_rag.paths import PDF_DIR

# Plain-language RAG topics shared by the open-access full-text harvesters.
RAG_TOPIC_QUERIES: list[str] = [
    "retrieval augmented generation",
    "dense passage retrieval",
    "hybrid retrieval reranking",
    "graphrag",
    "agentic rag",
    "rag evaluation",
    "query expansion retrieval",
    "long context rag",
    "chunking retrieval",
    "self-rag corrective rag",
]

# Topic-phrase -> topical tags (kept short; relevance gate does the real filtering).
_TOPIC_TAGS: dict[str, list[str]] = {
    "retrieval augmented generation": ["foundations"],
    "dense passage retrieval": ["dense-retrieval"],
    "hybrid retrieval reranking": ["hybrid-retrieval", "reranking"],
    "graphrag": ["graphrag"],
    "agentic rag": ["agentic"],
    "rag evaluation": ["evaluation"],
    "query expansion retrieval": ["query-expansion"],
    "long context rag": ["long-context"],
    "chunking retrieval": ["chunking"],
    "self-rag corrective rag": ["self-correcting"],
}

_MAX_OFFSET = 10_000


def _core_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("CORE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _authors(result: dict) -> list[str]:
    out: list[str] = []
    for a in result.get("authors") or []:
        name = (a.get("name") if isinstance(a, dict) else None) or ""
        name = name.strip()
        if name:
            out.append(name)
    return out


def _year(result: dict) -> int | None:
    year = result.get("yearPublished")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.strip().isdigit():
        return int(year.strip())
    return None


def _pdf_urls(result: dict) -> list[str]:
    urls: list[str] = []
    primary = result.get("downloadUrl")
    if isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    for u in result.get("sourceFulltextUrls") or []:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    ident = result.get("fullTextIdentifier")
    if isinstance(ident, str) and ident.strip():
        urls.append(ident.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _result_to_spec(result: dict, *, tags: list[str]) -> dict | None:
    title = (result.get("title") or "").strip()
    if not title:
        return None
    doi = (result.get("doi") or "").strip().lower() or None
    return {
        "doi": doi,
        "title": re.sub(r"\s+", " ", title),
        "abstract": (result.get("abstract") or "").strip() or None,
        "authors": _authors(result),
        "year": _year(result),
        "pdf_urls": _pdf_urls(result),
        "tags": tags,
    }


def _search_core(topic: str, *, max_results: int, limit: int = 100) -> list[dict]:
    log = get_logger()
    results: list[dict] = []
    offset = 0
    headers = _core_headers()
    while len(results) < max_results and offset + limit <= _MAX_OFFSET:
        params = {
            "q": f"{topic} _exists_:fullText",
            "limit": str(min(limit, max_results - len(results))),
            "offset": str(offset),
        }
        outcome = CORE.request("GET", CORE_SEARCH, params=params, headers=headers, timeout=60.0)
        if outcome.kind is not OutcomeKind.SUCCESS:
            log.warning("core search %r stopped: %s", topic, outcome.kind.value)
            break
        data = outcome.data or {}
        page = data.get("results") or []
        if not page:
            break
        results.extend(page)
        offset += len(page)
        if len(page) < limit:
            break
    return results[:max_results]


def harvest_core(
    *,
    max_works: int = 200,
    dry_run: bool = False,
    workers: int | None = None,
    existing_ids: set[str] | None = None,
) -> list[ManifestItem]:
    log = get_logger()
    skip_ids = existing_ids or set()
    by_doi: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    per_topic = max(20, max_works // max(1, len(RAG_TOPIC_QUERIES)))

    try:
        for topic in RAG_TOPIC_QUERIES:
            if len(by_id) >= max_works:
                break
            tags = _TOPIC_TAGS.get(topic, [])
            try:
                results = _search_core(topic, max_results=per_topic)
            except Exception as exc:
                log.warning("core search failed for %r: %s", topic, exc)
                continue
            log.info("core %s: %d candidate works", topic, len(results))
            for result in results:
                if len(by_id) >= max_works:
                    break
                spec = _result_to_spec(result, tags=tags)
                if not spec:
                    continue
                if not is_layout_relevant(spec["title"], spec["abstract"]):
                    continue
                key = spec["doi"] or slug_id(spec["title"])
                doc_id = slug_id(f"core-{key}")
                if doc_id in skip_ids or doc_id in by_id:
                    continue
                if spec["doi"] and spec["doi"] in by_doi:
                    continue
                spec["id"] = doc_id
                by_id[doc_id] = spec
                if spec["doi"]:
                    by_doi[spec["doi"]] = spec
    except Exception as exc:
        log.warning("core harvest discovery failed: %s", exc)

    def _finalize(spec: dict) -> ManifestItem:
        item = ManifestItem(
            id=spec["id"],
            title=spec["title"],
            authors=spec.get("authors", []),
            year=spec.get("year"),
            source="core",
            url=(spec["pdf_urls"][0] if spec["pdf_urls"] else (f"https://doi.org/{spec['doi']}" if spec["doi"] else "")),
            localPath=None,
            contentType="text/metadata",
            status="metadata_only" if (spec["title"] and spec["doi"]) else "failed",
            tags=sorted({*spec["tags"], "core", "open-access"}),
            doi=spec.get("doi"),
            abstract=spec.get("abstract"),
        )
        if dry_run:
            return item
        dest = PDF_DIR / f"{spec['id']}.pdf"
        for url in spec["pdf_urls"]:
            try:
                dl = download_to_file(dest, url, doc_id=spec["id"], doi=spec.get("doi"), stage="core")
                if dl.get("ok") and dest.exists() and dest.read_bytes()[:4] == b"%PDF":
                    item.status = "ok"
                    item.url = url
                    item.sha256 = dl.get("sha256")
                    item.contentType = "application/pdf"
                    item.localPath = relative_local_path(dest)
                    return item
                dest.unlink(missing_ok=True)
            except Exception as exc:
                log.debug("core download failed for %s: %s", url[:120], exc)
                dest.unlink(missing_ok=True)
        return item

    return parallel_map(_finalize, list(by_id.values()), workers=workers, label="core")
