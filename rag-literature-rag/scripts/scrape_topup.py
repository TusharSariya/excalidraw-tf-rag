"""Top-up scrape: broaden the topic-query set to add more net-new OA full-text.

The first pass (scrape_overnight.py) was candidate-bound by a 10-query set.
This monkeypatches a wider, non-overlapping RAG/IR subtopic query list onto the
two productive sources (s2orc, europepmc) and reruns; existing_ids dedup means
only genuinely new papers are added. doab (near-zero yield) and core (no key)
are skipped.
"""
from __future__ import annotations

import sys
import traceback

from rag_literature_rag.harvest import pmc as pmc_mod
from rag_literature_rag.harvest import s2orc as s2orc_mod
from rag_literature_rag.harvest.download import set_download_limit
from rag_literature_rag.harvest.ledger import init_db
from rag_literature_rag.harvest.log import setup_harvest_logging
from rag_literature_rag.harvest.parallel import set_workers
from rag_literature_rag.harvest.verify import verify_manifest
from rag_literature_rag.manifest import load_manifest, save_manifest, upsert_item

WORKERS = 16

BROAD_QUERIES = [
    "dense retrieval contrastive training",
    "passage reranking cross-encoder",
    "colbert late interaction retrieval",
    "splade sparse lexical retrieval",
    "hypothetical document embeddings hyde",
    "reciprocal rank fusion retrieval",
    "query rewriting for retrieval",
    "multi-hop question answering retrieval",
    "knowledge graph retrieval augmented generation",
    "retrieval augmented language model pretraining",
    "fusion-in-decoder open domain question answering",
    "in-context retrieval augmented language models",
    "retrieval augmented generation hallucination",
    "retrieval augmented generation faithfulness attribution",
    "embedding model fine-tuning retrieval",
    "instruction-tuned text embeddings",
    "approximate nearest neighbor vector search",
    "semantic chunking document segmentation",
    "passage retrieval benchmark beir",
    "retrieval evaluation metrics",
    "multi-vector dense retrieval",
    "learned sparse retrieval",
    "conversational retrieval augmented generation",
    "tool augmented language model retrieval",
    "retrieval augmented generation survey",
    "open domain question answering retriever reader",
    "long document retrieval",
    "retrieval augmented generation reasoning",
    "adaptive retrieval augmented generation",
]


def _ok(m) -> int:
    return sum(1 for i in m.items if i.status == "ok")


def main() -> int:
    log = setup_harvest_logging(verbose=True)
    init_db()
    set_workers(WORKERS)
    set_download_limit(WORKERS)

    # Broaden both productive sources' query lists.
    for mod in (s2orc_mod, pmc_mod):
        if hasattr(mod, "RAG_TOPIC_QUERIES"):
            mod.RAG_TOPIC_QUERIES = BROAD_QUERIES

    manifest = load_manifest()
    before_total, before_ok = len(manifest.items), _ok(manifest)
    log.info("topup start: total=%d ok=%d", before_total, before_ok)

    for name, fn, cap in [("s2orc", s2orc_mod.harvest_s2orc, 600), ("europepmc", pmc_mod.harvest_pmc, 600)]:
        try:
            existing = {i.id for i in manifest.items}
            items = fn(max_works=cap, existing_ids=existing, workers=WORKERS)
            added_ok = sum(1 for it in items if it.status == "ok")
            for it in items:
                upsert_item(manifest, it)
            save_manifest(manifest)
            log.info("%s: fetched=%d ok_in_batch=%d running_ok=%d", name, len(items), added_ok, _ok(manifest))
            print(f"[{name}] fetched={len(items)} ok_in_batch={added_ok} running_ok={_ok(manifest)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("%s FAILED: %s", name, exc)
            traceback.print_exc()

    try:
        verify_manifest(manifest, downgrade=True)
        save_manifest(manifest)
    except Exception as exc:  # noqa: BLE001
        log.exception("verify failed: %s", exc)

    after_total, after_ok = len(manifest.items), _ok(manifest)
    summary = f"TOPUP DONE total {before_total}->{after_total} | ok {before_ok}->{after_ok} (+{after_ok-before_ok})"
    log.info(summary)
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
