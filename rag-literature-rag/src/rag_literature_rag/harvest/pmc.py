"""Europe PMC open-access full-text harvest (REST search).

Most biomedical hits will be dropped by the RAG relevance gate — that is expected;
this source is here for the rare retrieval/IR papers indexed in Europe PMC.
"""

from __future__ import annotations

import re

from rag_literature_rag.harvest.download import download_to_file
from rag_literature_rag.harvest.http_client import get_json
from rag_literature_rag.harvest.log import get_logger
from rag_literature_rag.harvest.parallel import parallel_map
from rag_literature_rag.harvest.relevance import is_layout_relevant
from rag_literature_rag.manifest import ManifestItem, relative_local_path, slug_id
from rag_literature_rag.paths import PDF_DIR

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

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


def _authors(result: dict) -> list[str]:
    author_string = (result.get("authorString") or "").strip()
    if author_string:
        return [a.strip() for a in author_string.split(",") if a.strip()]
    out: list[str] = []
    for a in (result.get("authorList") or {}).get("author") or []:
        name = (a.get("fullName") if isinstance(a, dict) else None) or ""
        name = name.strip()
        if name:
            out.append(name)
    return out


def _year(result: dict) -> int | None:
    year = result.get("pubYear")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.strip().isdigit():
        return int(year.strip())
    return None


def _pdf_url(result: dict) -> str | None:
    full = (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    for entry in full:
        if not isinstance(entry, dict):
            continue
        if entry.get("documentStyle") == "pdf" and entry.get("availability") == "Open access":
            url = (entry.get("url") or "").strip()
            if url:
                return url
    return None


def _result_to_spec(result: dict, *, tags: list[str]) -> dict | None:
    title = (result.get("title") or "").strip()
    if not title:
        return None
    doi = (result.get("doi") or "").strip().lower() or None
    return {
        "doi": doi,
        "title": re.sub(r"\s+", " ", title),
        "abstract": (result.get("abstractText") or "").strip() or None,
        "authors": _authors(result),
        "year": _year(result),
        "pmcid": (result.get("pmcid") or "").strip() or None,
        "pdf_url": _pdf_url(result),
        "tags": tags,
    }


def _search_pmc(topic: str, *, max_results: int) -> list[dict]:
    log = get_logger()
    results: list[dict] = []
    cursor = "*"
    while len(results) < max_results:
        params = {
            "query": f"{topic} AND OPEN_ACCESS:Y AND IN_EPMC:Y",
            "resultType": "core",
            "format": "json",
            "pageSize": "1000",
            "cursorMark": cursor,
        }
        data = get_json(EUROPE_PMC_SEARCH, params=params, timeout=60.0)
        if not data:
            log.warning("europepmc search %r returned nothing", topic)
            break
        page = (data.get("resultList") or {}).get("result") or []
        if not page:
            break
        results.extend(page)
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return results[:max_results]


def harvest_pmc(
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
                results = _search_pmc(topic, max_results=per_topic)
            except Exception as exc:
                log.warning("europepmc search failed for %r: %s", topic, exc)
                continue
            log.info("europepmc %s: %d candidate works", topic, len(results))
            for result in results:
                if len(by_id) >= max_works:
                    break
                spec = _result_to_spec(result, tags=tags)
                if not spec:
                    continue
                if not is_layout_relevant(spec["title"], spec["abstract"]):
                    continue
                key = spec["doi"] or spec["pmcid"] or slug_id(spec["title"])
                doc_id = slug_id(f"europepmc-{key}")
                if doc_id in skip_ids or doc_id in by_id:
                    continue
                if spec["doi"] and spec["doi"] in by_doi:
                    continue
                spec["id"] = doc_id
                by_id[doc_id] = spec
                if spec["doi"]:
                    by_doi[spec["doi"]] = spec
    except Exception as exc:
        log.warning("europepmc harvest discovery failed: %s", exc)

    def _finalize(spec: dict) -> ManifestItem:
        item = ManifestItem(
            id=spec["id"],
            title=spec["title"],
            authors=spec.get("authors", []),
            year=spec.get("year"),
            source="europepmc",
            url=spec["pdf_url"] or (f"https://doi.org/{spec['doi']}" if spec["doi"] else ""),
            localPath=None,
            contentType="text/metadata",
            status="metadata_only" if (spec["title"] and spec["doi"]) else "failed",
            tags=sorted({*spec["tags"], "europepmc", "pmc", "biomedical", "open-access"}),
            doi=spec.get("doi"),
            abstract=spec.get("abstract"),
        )
        if dry_run or not spec["pdf_url"]:
            return item
        dest = PDF_DIR / f"{spec['id']}.pdf"
        try:
            dl = download_to_file(
                dest, spec["pdf_url"], doc_id=spec["id"], doi=spec.get("doi"), stage="europepmc"
            )
            if dl.get("ok") and dest.exists() and dest.read_bytes()[:4] == b"%PDF":
                item.status = "ok"
                item.url = spec["pdf_url"]
                item.sha256 = dl.get("sha256")
                item.contentType = "application/pdf"
                item.localPath = relative_local_path(dest)
                return item
            dest.unlink(missing_ok=True)
        except Exception as exc:
            log.debug("europepmc download failed for %s: %s", spec["pdf_url"][:120], exc)
            dest.unlink(missing_ok=True)
        return item

    return parallel_map(_finalize, list(by_id.values()), workers=workers, label="europepmc")
