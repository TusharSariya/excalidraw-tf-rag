"""Proof script: compute single-chunk rate directly from the LanceDB index.

corpus-health's 'collapse' metric is blind to metadata-only docs (denominator
is PDF-backed docs only). This script gives the true picture across all docs.

Usage:
    uv run python scripts/single_chunk_rate.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lancedb  # noqa: E402

from graph_layout_rag.paths import PKG_ROOT  # noqa: E402

PROFILE = "cuda-qwen0.6b-1024"
DB_PATH = PKG_ROOT / "data" / "indexes" / PROFILE / "lancedb"


def main() -> None:
    db = lancedb.connect(str(DB_PATH))
    tbl = db.open_table("chunks")
    rows = tbl.search().select(["doc_id", "text"]).limit(200_000).to_list()

    cpd: Counter[str] = Counter()
    first_text: dict[str, str] = {}
    for row in rows:
        doc_id = row["doc_id"]
        cpd[doc_id] += 1
        if doc_id not in first_text:
            first_text[doc_id] = row["text"] or ""

    total = len(cpd)
    single = sum(1 for c in cpd.values() if c == 1)
    short = sum(1 for doc_id, c in cpd.items() if c == 1 and len(first_text[doc_id]) < 300)

    print(f"docs:         {total:,}")
    print(f"chunks:       {len(rows):,}")
    print(f"single-chunk: {single:,} ({100*single/total:.1f}%)")
    print(f"<300-char:    {short:,} ({100*short/total:.1f}%)")
    print(f"mean chunks/doc: {len(rows)/total:.1f}")


if __name__ == "__main__":
    main()
