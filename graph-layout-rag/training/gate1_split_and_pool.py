"""GATE 1 prep — split triples train/heldout (by query) + build a FIXED distractor pool.

Runs in the EVAL env (needs lancedb + the corpus). Produces, under the gate1 data dir:
  triples_train.jsonl      — triples for TRAINING queries (also feeds train_splade.py)
  triples_heldout.jsonl    — triples for HELD-OUT queries (UNSEEN in training)
  distractor_pool.jsonl    — {chunk_id, text} fixed random corpus sample, shared across
                             all queries + both models, EXCLUDING any chunk that is a
                             positive or hard-neg anywhere in the gate1 triples (so the
                             positive's rank is unambiguous and never double-counted).
  split_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent.parent
SRC = TOOL_ROOT / "src"
_RAG_COMMON_SRC = TOOL_ROOT.parent / "rag-common" / "src"
for _p in (SRC, _RAG_COMMON_SRC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = Path(os.getenv("GRAPH_RAG_TRAIN_DATA_DIR", str(TOOL_ROOT / "data" / "training")))


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_corpus_chunks() -> list[dict]:
    import lancedb

    from graph_layout_rag.paths import CHUNKS_TABLE, profile_index_paths

    profile = os.getenv("GRAPH_RAG_TRAIN_CORPUS_PROFILE", "cuda-qwen4b-1024")
    paths = profile_index_paths(profile)
    db = lancedb.connect(str(paths.lance_dir))
    rows = db.open_table(CHUNKS_TABLE).to_arrow().to_pylist()
    out = []
    for r in rows:
        t = (r.get("text") or "").strip()
        if len(t) < 200:
            continue
        out.append({"chunk_id": r["id"], "text": t})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-frac", type=float, default=0.13)
    ap.add_argument("--distractors", type=int, default=300,
                    help="fixed random distractor pool size (shared)")
    ap.add_argument("--text-chars", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    triples = _load_jsonl(DATA_DIR / "triples.jsonl")
    if not triples:
        print("FATAL: no triples", file=sys.stderr)
        return 1

    # split by UNIQUE query so a held-out query is fully unseen
    rng = random.Random(args.seed)
    by_q: dict[str, list[dict]] = {}
    for r in triples:
        by_q.setdefault(r["query"], []).append(r)
    queries = sorted(by_q)
    rng.shuffle(queries)
    n_hold = max(1, round(len(queries) * args.heldout_frac))
    hold_qs = set(queries[:n_hold])

    train_rows, hold_rows = [], []
    for q, rows in by_q.items():
        (hold_rows if q in hold_qs else train_rows).extend(rows)

    # collect all chunk ids that appear as positive or hard-neg ANYWHERE → exclude
    # from the distractor pool (keeps positive rank unambiguous).
    used_ids: set[str] = set()
    for r in triples:
        used_ids.add(r["positive_chunk_id"])
        used_ids.update(r.get("negative_chunk_ids") or [])

    corpus = _load_corpus_chunks()
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(corpus)
    distractors = []
    for c in corpus:
        if c["chunk_id"] in used_ids:
            continue
        distractors.append({"chunk_id": c["chunk_id"], "text": c["text"][: args.text_chars]})
        if len(distractors) >= args.distractors:
            break

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATA_DIR / "triples_train.jsonl", train_rows)
    _write_jsonl(DATA_DIR / "triples_heldout.jsonl", hold_rows)
    _write_jsonl(DATA_DIR / "distractor_pool.jsonl", distractors)

    summary = {
        "total_triples": len(triples),
        "unique_queries": len(queries),
        "heldout_queries": len(hold_qs),
        "train_queries": len(queries) - len(hold_qs),
        "train_rows": len(train_rows),
        "heldout_rows": len(hold_rows),
        "distractor_pool": len(distractors),
        "excluded_used_ids": len(used_ids),
        "heldout_frac": args.heldout_frac,
        "seed": args.seed,
    }
    (DATA_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print("SPLIT_SUMMARY:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
