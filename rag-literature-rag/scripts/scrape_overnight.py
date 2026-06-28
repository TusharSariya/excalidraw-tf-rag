"""Overnight bounded scrape of legal OA full-text from the working new sources.

Calls harvest_s2orc / harvest_pmc / harvest_doab directly (bypasses the full
discovery loop so no trusted-venues/OpenAlex noise), upserts into the manifest,
verifies, and reports the ok-PDF delta. CORE is skipped (needs CORE_API_KEY).

Run:
  uv run python scripts/scrape_overnight.py 2>&1 | tee data/scrape-overnight.log
"""
from __future__ import annotations

import logging
import sys
import traceback

from rag_literature_rag.harvest.doab import harvest_doab
from rag_literature_rag.harvest.download import set_download_limit
from rag_literature_rag.harvest.ledger import init_db
from rag_literature_rag.harvest.log import setup_harvest_logging
from rag_literature_rag.harvest.parallel import set_workers
from rag_literature_rag.harvest.pmc import harvest_pmc
from rag_literature_rag.harvest.s2orc import harvest_s2orc
from rag_literature_rag.harvest.verify import verify_manifest
from rag_literature_rag.manifest import load_manifest, save_manifest, upsert_item

WORKERS = 16
# Caps sized to net ~500 new ok PDFs after relevance gating + download success.
PLAN = [
    ("s2orc", harvest_s2orc, 500),
    ("europepmc", harvest_pmc, 400),
    ("doab", harvest_doab, 150),
]


def _ok(manifest) -> int:
    return sum(1 for i in manifest.items if i.status == "ok")


def main() -> int:
    log = setup_harvest_logging(verbose=True)
    init_db()
    set_workers(WORKERS)
    set_download_limit(WORKERS)

    manifest = load_manifest()
    before_total = len(manifest.items)
    before_ok = _ok(manifest)
    log.info("scrape start: total=%d ok=%d", before_total, before_ok)

    for name, fn, cap in PLAN:
        try:
            existing = {i.id for i in manifest.items}
            items = fn(max_works=cap, existing_ids=existing, workers=WORKERS)
            added_ok = 0
            for it in items:
                if it.status == "ok":
                    added_ok += 1
                upsert_item(manifest, it)
            save_manifest(manifest)
            log.info("%s: fetched=%d ok_in_batch=%d running_ok=%d", name, len(items), added_ok, _ok(manifest))
            print(f"[{name}] fetched={len(items)} ok_in_batch={added_ok} running_ok={_ok(manifest)}", flush=True)
        except Exception as exc:  # noqa: BLE001 — one source must not kill the run
            log.exception("%s FAILED: %s", name, exc)
            print(f"[{name}] ERROR {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    try:
        stats = verify_manifest(manifest, downgrade=True)
        save_manifest(manifest)
        log.info("verify: %s", stats)
    except Exception as exc:  # noqa: BLE001
        log.exception("verify failed: %s", exc)

    after_total = len(manifest.items)
    after_ok = _ok(manifest)
    summary = (
        f"DONE total {before_total}->{after_total} (+{after_total-before_total}) | "
        f"ok {before_ok}->{after_ok} (+{after_ok-before_ok})"
    )
    log.info(summary)
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
