# Research-Tool Campaign — 2026-07

Tracks execution of `~/.claude/plans/take-a-look-at-resilient-naur.md` (E0 preconditions +
E1 knowledge-map layer + E1 eval gate) on `research-tool-e0-e1`. Companion doc:
`docs/research-tool-decisions-log-2026-07.md` (chronological decision record).

## Status

- **Workstream A (E0 preconditions): mostly complete.** `accelerate` dep fix landed (4-bit
  loads correctly again). Citation-graph coverage audit (`cite coverage`) + arXiv-ID/title
  keying upgrade shipped for both tools. GLR and RLR `cite enrich` jobs have been running in
  the background throughout this campaign — **not yet complete as of this doc**; the citation
  graphs used below reflect whatever coverage those jobs had reached at measurement time
  (see "Pending" below).
- **Workstream B (E1 litmap core + `map` CLI): complete.** `rag_common.litmap/` (doc_vectors,
  similarity/RRF fusion, communities/Leiden, mainpath/SPC, neighborhood) + `map
  clusters|lineage|similar|seminal` wired into both tools' `cli.py`. Offline unit tests green.
- **Workstream C (E1 eval gate): complete, this doc's subject.** `rag_common/evals/`
  (concentration, ablation, survey_gold, judge_calibration) + `eval map-gate` / `eval
  judge-calibrate` CLI wiring in both tools + this pinned regression floor.

## C1 — `rag_common/evals/` (shared library)

- `concentration.py` — Herfindahl-index reference-concentration score per survey/section,
  compared against uniform-random and size-matched-random null partitions (n=100 draws,
  mean/std/z-score).
- `ablation.py` — hybrid (alpha=0.5) / text-only (alpha=0) / citation-only (alpha=1)
  clustering ablation over `rag_common.litmap`, scored by `concentration.py`; promotion gate
  is PROMOTE iff hybrid's mean size-matched z-score beats **both** single-signal arms.
- `survey_gold.py` — survey gold-set builder from a tool's `citations.sqlite` + manifest:
  keyword match (`survey|state of the art|taxonomy`, `handbook|monograph|chapter` as an
  opt-in fallback) OR >=100 in-corpus outgoing references. Doc-level granularity now (single
  `"_all"` section); per-section is a documented future seam, not implemented (needs
  section-scoped citation-context extraction the ingest chunk store doesn't carry yet).
- `judge_calibration.py` — labeling-sheet builder/parser + Cohen's kappa agreement scoring +
  decisions-log-item-19 fail-path classification (PASS / REVISE_AND_RETRY / HUMAN_JUDGED).

33 new offline unit tests in `rag-common/tests/test_evals_{concentration,ablation,
survey_gold,judge_calibration}.py` — synthetic partitions with known concentration, a
size-distribution-preservation check on the size-matched null, gate PROMOTE/BLOCK logic
(including tie and missing-arm cases), survey detection against a fixture sqlite DB, kappa
against hand-computed values, and a labeling-sheet round trip. All offline tests in
`rag-common` (152 total) pass; `uvx ruff check`/`format` clean.

## C2 — CLI wiring

Both tools get `eval map-gate [--json] [--profile ...] [--min-refs 100] [--fallback-keywords]`
and `eval judge-calibrate [--build-sheet | --score <sheet>]`, mirrored via
`eval/map_gate.py` in each tool (thin adapter over `rag_common.evals` + each tool's
`query/litmap_support.py`/`citation_store.py`/`manifest.py` — same seam as `map_gate.py`'s
sibling `query/litmap_cli.py`).

`map-gate` writes `data/eval/runs/<ts>-map-gate/report.{json,md}` and degrades gracefully
with no citation store at all (text_only-only arm, gate BLOCKs with a named reason) rather
than crashing — verified by running it against RLR before its citation graph existed.

### Provisional `map-gate` run (2026-07-02, both tools — NOT the E1 promotion decision)

These are smoke/dry-run numbers taken **during** the campaign, with `cite enrich` still
running in the background on both tools and `--n-random` set low (5, not the CLI default of
100) purely to keep the smoke run fast — re-run with `--n-random 100` (default) once both
`cite enrich` jobs are complete for the real E1 gate decision.

| Tool | doc_count | qualifying surveys (min_refs=100) | citation signal | hybrid z(size-matched) | text_only z | citation_only z | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| GLR (`cuda-qwen0.6b-1024`) | 5,819 | 31 | yes | 36.567 | 34.575 | 51.383 | BLOCK (loses to citation_only) |
| RLR (`cuda-qwen0.6b-1024`) | 1,270 | 35 | yes | 5.490 | 7.872 | 10.062 | BLOCK (loses to both) |

Both runs completed **end-to-end with no crash** — the C2 acceptance bar ("must run
end-to-end on GLR's current data without crashing; RLR numbers are provisional"). Neither
currently clears the E1 promotion gate: on both corpora, the pure citation-coupling signal
alone concentrates survey references better than the RRF-fused hybrid at the default
`alpha=0.5`/`top_k=10`/Leiden-resolution=1.0 settings used here. This is a real (if
preliminary) finding, not a bug — plausible next steps (out of scope for this workstream,
noted for the E1 gate run once `cite enrich` completes): sweep `alpha` above 0.5 (citation
signal may just need more weight in the fusion, not to win outright at parity), and re-check
after keying coverage improves (both citation graphs were still mid-enrichment when this was
measured, and `citation_only`'s edge here may partly reflect strong coupling within the
currently-keyed subset rather than the true corpus).

## C3 — Pinned regression floor (`scripts/regression-floor.sh`)

Pinned to `cuda-qwen0.6b-1024` on both tools (the index that exists on this machine;
`contextual-v1` has no index here — decisions-log item 28). GLR runs `hybrid` on `catalog` +
`catalog-nl` (a de-biased qrels overlay on the `catalog` track, not a separate `EvalTrack`);
RLR runs `hybrid` on `catalog` + `pdf-deep-read`. No `--dense-weight`/`--sparse-weight` CLI
flags exist on `eval benchmark` — the plain `hybrid` strategy already resolves to each tool's
`query/hybrid.py` module constants, which match the plan's target weights exactly (GLR
dw=1.0/sw=2.0, RLR dw=1.0/sw=0.3 — see the script's header comment for the full mapping
rationale). Tolerance: **+/-0.005 nDCG@10** per (tool, track) on every E1 merge.

### Baseline (recorded 2026-07-02)

Full JSON/markdown reports:
- GLR catalog: `graph-layout-rag/data/eval/runs/20260702T151628Z-cuda-qwen0.6b-1024-9831cd73/`
- GLR catalog-nl: `graph-layout-rag/data/eval/runs/20260702T151651Z-cuda-qwen0.6b-1024-28157bd0/`
- RLR catalog: `rag-literature-rag/data/eval/runs/20260702T151716Z-cuda-qwen0.6b-1024-7b023d9c/`
- RLR pdf-deep-read: `rag-literature-rag/data/eval/runs/20260702T151732Z-cuda-qwen0.6b-1024-efdd8638/`

| Tool | Track | Strategy | nDCG@10 | MRR | Recall@10 | HR@5 | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GLR | catalog | hybrid | 0.8619 | 0.9456 | 0.9116 | 1.000 | 0 |
| GLR | catalog-nl | hybrid | 0.7784 | 0.9796 | 0.3273 | 1.000 | 0 |
| RLR | catalog | hybrid | 0.5315 | 0.4162 | 0.8929 | 0.881 | 4 |
| RLR | pdf-deep-read | hybrid | 0.5327 | 0.4104 | 0.9146 | 0.878 | 3 |

**Deviation noted, not fixed (out of scope for this workstream):** every run above logged
`4-bit load failed for Qwen/Qwen3-Embedding-0.6B ... falling back to FP16` — the venvs this
ran in do not actually have `bitsandbytes` installed (`accelerate` is in `rag-common`'s base
deps per Workstream A's A1, but the tool venvs weren't re-synced with `--extra
retrieval-experiments-gpu --extra dev` before this run). A1's acceptance bar ("a query run
no longer prints the 4-bit fallback warning") is therefore not yet met in these venvs; the
numbers above are still valid FP16 nDCG@10 baselines (same regression-floor purpose either
way), but re-running `uv sync --extra retrieval-experiments-gpu --extra dev` in each tool
before the *next* regression-floor run would both fix the warning and is worth a quick
re-check that FP16 vs 4-bit doesn't shift these numbers outside the +/-0.005 tolerance.

**RLR's ~0.53 nDCG@10 vs. the plan's cited "RLR 0.942 nDCG@10"**: this floor run uses the
plain hand-curated gold set with no `--qrels` overlay (as specified — the plan's A3 command
has no `--qrels` flag), which is the "raw nDCG@10 (hole-distorted)" number in RLR's own
quality-campaign docs (`rag-literature-rag/docs/quality-campaign-2026-06-27.md` records raw
catalog nDCG@10 of 0.568 pre-4B, in the same ballpark as 0.531 here) — hole-distorted because
many actually-relevant docs are unlabeled in the curated set and score as misses. The 0.942
figure quoted in the plan's context section is very likely a corrected/condensed metric from
a qrels-overlaid run, not directly comparable to this floor's raw number. Not a discrepancy
to resolve here — the pinned floor intentionally uses the exact command in decisions-log
item 28/A3, and the +/-0.005 tolerance applies to *this* number on future runs of the same
command, not to reconciling it with the plan's headline figure.

## Pending

- **E1 gate run (real, non-provisional):** awaiting both tools' `cite enrich` background
  jobs to complete, plus a `cite coverage` re-run to confirm the >=80% keyed-doc target
  (Workstream A's A2 acceptance bar) before the map-gate numbers above are trustworthy. Once
  both graphs are re-keyed, re-run `eval map-gate` (default `--n-random 100`) on both tools
  and update the provisional table above with the real decision.
- **Judge calibration labeling sheet:** `eval judge-calibrate --build-sheet` was smoke-tested
  on both tools (cluster-label-sanity slice — lineage/faithfulness calibration needs E1
  lineage output + the user and is explicitly out of scope for this workstream). The
  generated sheets (`data/eval/judge_calibration/cluster_label_sheet.md` in each tool) are
  awaiting the user's hand labels before `--score` can report a real kappa.
- **Alpha sweep on the ablation gate:** see the provisional-run note above — worth a
  dedicated follow-up once the citation graphs are fully keyed, before concluding E1's
  hybrid clustering doesn't beat citation-only.
