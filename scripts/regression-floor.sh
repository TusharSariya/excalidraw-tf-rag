#!/usr/bin/env bash
# Pinned regression floor for the E1 knowledge-map campaign (Workstream A3 / C3).
#
# Runs a fixed, GPU-eval benchmark on each tool and prints the resulting nDCG@10 numbers.
# Every E1 merge should stay within +/-0.005 nDCG@10 of the last recorded baseline for each
# (tool, track) pair below (see docs/research-tool-campaign-2026-07.md for the recorded
# numbers) — a bigger swing means something in the shared litmap/eval code path silently
# touched retrieval, which is out of scope for this campaign (NOT in scope: any
# retrieval-ranking change, decisions-log item 22/plan "NOT in scope").
#
# Pinned commands (both tools use `cuda-qwen0.6b-1024` — the index that actually exists on
# this machine; `contextual-v1` has no index here, see decisions-log item 28):
#
#   GLR: eval benchmark --embed-profile cuda-qwen0.6b-1024 --strategy hybrid \
#        --track catalog [--qrels data/eval/qrels/catalog-nl/qrels.json for the -nl track]
#   RLR: eval benchmark --embed-profile cuda-qwen0.6b-1024 --strategy hybrid \
#        --track catalog | --track pdf-deep-read
#
# Weight mapping (plan asked for GLR hybrid dw=1.0/sw=2.0 and RLR hybrid sw=0.3): NO extra
# flags are needed. `eval benchmark` has no --dense-weight/--sparse-weight flags; the plain
# `hybrid` strategy already resolves to each tool's `query/hybrid.py` module constants,
# which are exactly the plan's target weights on both tools as of this writing:
#   - graph-layout-rag/src/graph_layout_rag/query/hybrid.py: DENSE_WEIGHT=1.0, SPARSE_WEIGHT=2.0
#   - rag-literature-rag/src/rag_literature_rag/query/hybrid.py: DENSE_WEIGHT=1.0, SPARSE_WEIGHT=0.3
# If those constants ever change, this script's "hybrid" strategy silently follows them —
# that's intentional (the floor tracks *production* hybrid, not a frozen weight pair), but
# it means a weight-tuning campaign should re-run this script and update the baseline table
# in docs/research-tool-campaign-2026-07.md rather than assuming the old numbers still hold.
#
# `catalog-nl` is not a distinct EvalTrack (`EvalTrack` is only "catalog"/"pdf-deep-read")
# — it is a de-biased qrels *overlay* applied on top of the `catalog` track's gold cases
# (case ids match 1:1; see graph-layout-rag/data/eval/qrels/catalog-nl/qrels.json), so it's
# invoked as `--track catalog --qrels data/eval/qrels/catalog-nl/qrels.json`.
#
# Tolerance: +/-0.005 nDCG@10 per (tool, track) vs. the recorded baseline.
#
# Runs GLR then RLR *sequentially*, never in parallel — both tools share one 8GB 3060 Ti and
# a concurrent run would OOM or silently corrupt timing numbers.
#
# Usage: scripts/regression-floor.sh [--json-only]
#   (no args): human-readable summary to stdout, full JSON+markdown reports on disk under
#   each tool's data/eval/runs/<ts>-<profile>-<id>/ (per eval benchmark's own convention).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="cuda-qwen0.6b-1024"

echo "== regression floor: graph-layout-rag =="
cd "$REPO_ROOT/graph-layout-rag"

echo "-- GLR catalog --"
uv run graph-layout-rag eval benchmark \
  --embed-profile "$PROFILE" \
  --strategy hybrid \
  --track catalog \
  --report -v

echo "-- GLR catalog-nl (qrels overlay on the catalog track) --"
uv run graph-layout-rag eval benchmark \
  --embed-profile "$PROFILE" \
  --strategy hybrid \
  --track catalog \
  --qrels data/eval/qrels/catalog-nl/qrels.json \
  --report -v

echo "== regression floor: rag-literature-rag =="
cd "$REPO_ROOT/rag-literature-rag"

echo "-- RLR catalog --"
uv run rag-literature-rag eval benchmark \
  --embed-profile "$PROFILE" \
  --strategy hybrid \
  --track catalog \
  --report -v

echo "-- RLR pdf-deep-read --"
uv run rag-literature-rag eval benchmark \
  --embed-profile "$PROFILE" \
  --strategy hybrid \
  --track pdf-deep-read \
  --report -v

echo "== done. Check data/eval/runs/<ts>-*/benchmark.json + report.md in each tool for full numbers. =="
