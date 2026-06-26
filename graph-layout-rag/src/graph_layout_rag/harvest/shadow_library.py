"""Shadow-library PDF resolver — open-access APIs first, Patchright fallback.

Fetch strategy (tried in order, short-circuits on first hit):
  1. Unpaywall API   — legal best-OA URL for any DOI (covers arXiv, LIPIcs, CEUR-WS,
                       DMTCS, JGAA, publisher OA, institutional repos). Pure httpx, ~0.5s.
  2. S2 openAccessPdf — Semantic Scholar's OA PDF link. httpx, ~0.5s.
  3. sci-hub.{ru,se,st} — Patchright Chromium, citation_pdf_url scrape.
  4. Anna's Archive {gl,vg,pk,gd} → LibGen+ {li,bz,la,gl,vg} — MD5 chain.

Manifest is checkpointed every CHECKPOINT_EVERY items so crashes lose at most
that many successful fetches.

CLI:
    uv run graph-layout-rag harvest shadow-fetch [--workers N] [--limit N] [--dry-run]
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from graph_layout_rag.harvest.log import get_logger
from graph_layout_rag.manifest import Manifest, ManifestItem, relative_local_path, save_manifest
from graph_layout_rag.paths import PDF_DIR

if TYPE_CHECKING:
    pass

SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.se",
    "https://sci-hub.st",
]

AA_MIRRORS = [
    "https://annas-archive.gl",
    "https://annas-archive.vg",
    "https://annas-archive.pk",
    "https://annas-archive.gd",
]

LIBGEN_MIRRORS = [
    "https://libgen.li",
    "https://libgen.bz",
    "https://libgen.la",
    "https://libgen.gl",
    "https://libgen.vg",
]

_TIMEOUT = 8_000          # ms per page navigation (fast fail on misses)
_LIBGEN_ADS_TIMEOUT = 12_000  # ms for libgen ads.php — needs networkidle for JS-rendered link
_DL_TIMEOUT = 30_000      # ms for browser file download (sci-hub) — real PDFs start in <5s
CHECKPOINT_EVERY = 25     # save manifest after every N items

# Unpaywall requires an email in the query string (free, no key)
_UNPAYWALL_EMAIL = "tushar.sariya77@gmail.com"


def _needs_pdf(item: ManifestItem) -> bool:
    return bool(item.doi) and item.status != "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _httpx_download(pdf_url: str, dest: Path, log, label: str) -> bool:
    """Download pdf_url with httpx; return True if a valid PDF lands at dest.

    Single-pass stream: writes to dest only after confirming %PDF in first chunk.
    Fails fast (<1s) for HTML/error responses; no fixed size timeout for real PDFs.
    """
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with httpx.stream(
            "GET", pdf_url, follow_redirects=True,
            timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5),
            headers={"User-Agent": "Mozilla/5.0 (compatible; graph-layout-rag)"},
        ) as resp:
            if resp.status_code != 200:
                log.debug("shadow-fetch %s http %d", label, resp.status_code)
                return False
            header_buf = b""
            checked = False
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    if not checked:
                        header_buf += chunk
                        if len(header_buf) >= 8:
                            if header_buf[:4] != b"%PDF":
                                log.debug("shadow-fetch %s non-PDF: %r", label, header_buf[:8])
                                return False
                            fh.write(header_buf)
                            checked = True
                    else:
                        fh.write(chunk)
            if not checked:
                log.debug("shadow-fetch %s empty response", label)
                return False
        tmp.rename(dest)
        size = dest.stat().st_size
        log.debug("shadow-fetch %s ok (httpx stream): %d bytes", label, size)
        return True
    except Exception as exc:
        log.debug("shadow-fetch %s httpx error: %s", label, exc)
    finally:
        tmp.unlink(missing_ok=True)
    return False


# ---------------------------------------------------------------------------
# Fast-path: no-browser open-access APIs
# ---------------------------------------------------------------------------

def _fetch_unpaywall(doi: str, dest: Path, log) -> bool:
    """Unpaywall best-OA lookup — covers arXiv, LIPIcs, CEUR-WS, publisher OA."""
    import httpx
    try:
        resp = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": _UNPAYWALL_EMAIL},
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        loc = data.get("best_oa_location") or {}
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            return False
        return _httpx_download(pdf_url, dest, log, f"unpaywall({doi[:30]})")
    except Exception as exc:
        log.debug("shadow-fetch unpaywall fail %s: %s", doi, exc)
        return False


def _fetch_s2_oa(doi: str, dest: Path, log) -> bool:
    """Semantic Scholar openAccessPdf — catches arXiv preprints Unpaywall may miss."""
    import httpx
    try:
        resp = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        oa = data.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
        if not pdf_url:
            return False
        return _httpx_download(pdf_url, dest, log, f"s2-oa({doi[:30]})")
    except Exception as exc:
        log.debug("shadow-fetch s2-oa fail %s: %s", doi, exc)
        return False


# ---------------------------------------------------------------------------
# Browser path: sci-hub → Anna's Archive → LibGen (Patchright)
# ---------------------------------------------------------------------------

def _pdf_direct_fetch(page, pdf_url: str, dest: Path, log, label: str) -> bool:
    """Fetch pdf_url via httpx with browser cookies (Cloudflare clearance from patchright context).

    No browser-download fallback: page.expect_download blocks 30s+ when the PDF URL is
    Cloudflare-gated, even though httpx with the same cookies resolves it correctly.
    """
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cookies = {c["name"]: c["value"] for c in page.context.cookies()}
        resp = httpx.get(pdf_url, cookies=cookies, follow_redirects=True, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0", "Referer": page.url})
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            dest.write_bytes(resp.content)
            log.debug("shadow-fetch %s ok (httpx+cookies): %d bytes", label, len(resp.content))
            return True
        log.debug("shadow-fetch %s httpx non-PDF %d: %r", label, resp.status_code, resp.content[:8])
    except Exception as exc:
        log.debug("shadow-fetch %s httpx error: %s", label, exc)
    return False


def _fetch_scihub(page, doi: str, dest: Path, log) -> bool:
    """Try each sci-hub mirror in order. Returns True if a valid PDF was saved."""
    for base in SCIHUB_MIRRORS:
        try:
            page.goto(f"{base}/{doi}", timeout=_TIMEOUT, wait_until="domcontentloaded")
            html = page.content()

            m = re.search(r'citation_pdf_url[^>]*content="([^"]+)"', html)
            if not m:
                continue

            pdf_path = m.group(1)
            pdf_url = (base + pdf_path) if pdf_path.startswith("/") else pdf_path

            if _pdf_direct_fetch(page, pdf_url, dest, log, f"sci-hub({base})"):
                return True
        except Exception as e:
            log.debug("shadow-fetch sci-hub %s fail %s: %s", base, doi, e)
            dest.unlink(missing_ok=True)

    return False


def _search_aa_for_md5s(page, doi: str, log) -> list[str]:
    """Search Anna's Archive mirrors for MD5 hashes matching a DOI."""
    for base in AA_MIRRORS:
        try:
            page.goto(
                f"{base}/search?q=doi:{doi}&ext=pdf",
                timeout=_TIMEOUT,
                wait_until="domcontentloaded",
            )
            md5s = re.findall(r'href="/md5/([a-f0-9]{32})"', page.content())
            if md5s:
                log.debug("shadow-fetch AA %s found %d md5s for %s", base, len(md5s), doi)
                return md5s[:5]
        except Exception as e:
            log.debug("shadow-fetch AA %s fail %s: %s", base, doi, e)
    return []


def _download_libgen_md5(page, md5: str, dest: Path, log) -> bool:
    """Try LibGen+ mirrors: browser for ads.php (key extraction), httpx for actual download.

    ads.php renders the get.php link via JS, so we use networkidle (lightweight page,
    typically fires in <3s). Then httpx with browser cookies for the actual file download.
    """
    import httpx

    for base in LIBGEN_MIRRORS[:3]:  # top 3 most reliable mirrors
        try:
            # networkidle here: ads.php renders download link via JS (lightweight page)
            page.goto(
                f"{base}/ads.php?md5={md5}",
                timeout=_LIBGEN_ADS_TIMEOUT,
                wait_until="networkidle",
            )
            html = page.content()

            m = re.search(r'href="(get\.php\?md5=[a-f0-9]+&(?:amp;)?key=[A-Z0-9]+)"', html)
            if not m:
                continue

            rel = m.group(1).replace("&amp;", "&")
            dl_url = f"{base}/{rel}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            # httpx with browser cookies — key is session-bound on LibGen
            cookies = {c["name"]: c["value"] for c in page.context.cookies()}
            try:
                resp = httpx.get(
                    dl_url, cookies=cookies, follow_redirects=True, timeout=45,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                    dest.write_bytes(resp.content)
                    log.debug("shadow-fetch libgen %s ok md5=%s (%d bytes)", base, md5, len(resp.content))
                    return True
                log.debug("shadow-fetch libgen %s non-PDF md5=%s status=%d fmt=%r", base, md5, resp.status_code, resp.content[:8])
            except Exception as dl_exc:
                log.debug("shadow-fetch libgen %s dl fail md5=%s: %s", base, md5, dl_exc)
            dest.unlink(missing_ok=True)
        except Exception as e:
            log.debug("shadow-fetch libgen %s fail md5=%s: %s", base, md5, e)
            dest.unlink(missing_ok=True)

    return False


def _fetch_aa_libgen(page, doi: str, dest: Path, log) -> bool:
    """Search Anna's Archive mirrors, then try each MD5 on all LibGen+ mirrors."""
    md5s = _search_aa_for_md5s(page, doi, log)
    if not md5s:
        return False
    for md5 in md5s:
        if _download_libgen_md5(page, md5, dest, log):
            return True
    return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _worker(
    queue: "Queue[tuple[int, ManifestItem, int]]",
    headless: bool,
    manifest_items: list[ManifestItem],
    lock: threading.Lock,
    counters: dict[str, int],
    source_counts: dict[str, int],
    log,
) -> None:
    """One parallel shadow-fetch worker — owns its own Playwright + Chromium process + page."""
    from patchright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        try:
            while True:
                try:
                    entry = queue.get(timeout=2)
                except Empty:
                    break

                idx, item, total = entry
                doi = item.doi or ""
                dest = PDF_DIR / f"{item.id}.pdf"
                log.info("shadow-fetch [%d/%d] %s", idx + 1, total, doi)

                source = None
                ok = _fetch_unpaywall(doi, dest, log)
                if ok:
                    source = "unpaywall"
                if not ok:
                    ok = _fetch_s2_oa(doi, dest, log)
                    if ok:
                        source = "s2-oa"
                if not ok:
                    ok = _fetch_scihub(page, doi, dest, log)
                    if ok:
                        source = "scihub"
                if not ok:
                    ok = _fetch_aa_libgen(page, doi, dest, log)
                    if ok:
                        source = "aa-libgen"

                with lock:
                    if ok and dest.exists():
                        sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
                        item.status = "ok"
                        item.localPath = relative_local_path(dest)
                        item.sha256 = sha256
                        counters["fetched"] += 1
                        source_counts[source] = source_counts.get(source, 0) + 1
                        log.info("shadow-fetch ok [%s]: %s → %s", source, doi, dest.name)
                    else:
                        counters["failed"] += 1
                    counters["processed"] += 1
                    processed = counters["processed"]
                    fetched = counters["fetched"]
                    failed = counters["failed"]

                if processed % CHECKPOINT_EVERY == 0:
                    with lock:
                        save_manifest(Manifest(items=manifest_items))
                    log.info(
                        "shadow-fetch checkpoint: fetched=%d failed=%d (%d/%d) sources=%s",
                        fetched, failed, processed, total, dict(source_counts),
                    )

                time.sleep(0.3)
        finally:
            ctx.close()
            browser.close()


def shadow_fetch_manifest(
    manifest_items: list[ManifestItem],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    headless: bool = True,
    workers: int = 1,
) -> dict[str, int]:
    """Fetch PDFs for manifest items with DOI but no local PDF.

    Mutates items in place (status, localPath, sha256). Checkpoints manifest to
    disk every CHECKPOINT_EVERY items so crashes don't lose progress.
    Returns {candidates, fetched, failed, skipped}.

    workers > 1 launches N independent Chromium processes in parallel threads.
    """
    log = get_logger()
    candidates = [i for i in manifest_items if _needs_pdf(i)]
    if limit:
        candidates = candidates[:limit]

    log.info(
        "shadow-fetch: %d candidates | workers=%d | unpaywall+s2(httpx) → sci-hub(%d) → AA(%d)→libgen(%d)",
        len(candidates), workers, len(SCIHUB_MIRRORS), len(AA_MIRRORS), len(LIBGEN_MIRRORS),
    )
    if dry_run:
        return {"candidates": len(candidates), "fetched": 0, "failed": 0, "skipped": len(candidates)}

    counters: dict[str, int] = {"fetched": 0, "failed": 0, "processed": 0}
    source_counts: dict[str, int] = {}
    lock = threading.Lock()

    queue: Queue = Queue()
    for i, item in enumerate(candidates):
        queue.put((i, item, len(candidates)))

    threads = [
        threading.Thread(
            target=_worker,
            args=(queue, headless, manifest_items, lock, counters, source_counts, log),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    save_manifest(Manifest(items=manifest_items))
    log.info(
        "shadow-fetch done: fetched=%d failed=%d / %d candidates | sources=%s",
        counters["fetched"], counters["failed"], len(candidates), source_counts,
    )
    return {"candidates": len(candidates), "fetched": counters["fetched"], "failed": counters["failed"], "skipped": 0}
