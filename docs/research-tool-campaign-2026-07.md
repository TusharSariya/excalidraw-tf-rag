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
- **Workstream C (E1 eval gate): complete, hardened.** `rag_common/evals/` (concentration,
  ablation, survey_gold, judge_calibration) + `eval map-gate` / `eval judge-calibrate` CLI
  wiring in both tools + this pinned regression floor. A follow-up hardening pass (this doc's
  current subject) replaced the raw cross-arm z-score comparison — which BLOCKed both tools
  on a granularity-mismatch artifact (see "Provisional run superseded" below) — with a
  resolution-swept, common-support, paired-CI gate. Both tools now gate to **TIE**.

## C1 — `rag_common/evals/` (shared library)

- `concentration.py` — Herfindahl-index reference-concentration score per survey/section,
  compared against uniform-random and size-matched-random null partitions (n=100 draws,
  mean/std/z-score); `cluster_coverage`/`cluster_count_min2` (hardening pass) — granularity
  and corpus-coverage diagnostics computed over the whole partition, not just gold refs.
- `ablation.py` — hybrid (alpha=0.5) / text-only (alpha=0) / citation-only (alpha=1)
  clustering ablation over `rag_common.litmap`, scored by `concentration.py`. Hardening pass:
  every arm is resolution-swept to a shared, granularity-matched `target_k` (`--target-k`
  overrides; `ALPHA_SWEEP_VARIANTS`/`--alpha-sweep` adds alpha in {0.25, 0.75}), surveys are
  scored on **common support** (refs every compared arm actually clustered), and the
  promotion gate (`PROMOTE`/`TIE`/`BLOCK`) is a paired bootstrap-CI + sign-test comparison of
  per-survey score differences (hybrid vs. each other arm), not a bare z-score comparison —
  see the module docstring for the full rationale and the mechanism the original provisional
  BLOCK verdicts turned out to be measuring (cluster-count-mismatch, not a real signal).
- `survey_gold.py` — survey gold-set builder from a tool's `citations.sqlite` + manifest:
  keyword match (`survey|state of the art|taxonomy`, `handbook|monograph|chapter` as an
  opt-in fallback) OR >=100 in-corpus outgoing references. Doc-level granularity now (single
  `"_all"` section); per-section is a documented future seam, not implemented (needs
  section-scoped citation-context extraction the ingest chunk store doesn't carry yet).
- `judge_calibration.py` — labeling-sheet builder/parser + Cohen's kappa agreement scoring +
  decisions-log-item-19 fail-path classification (PASS / REVISE_AND_RETRY / HUMAN_JUDGED).

50 offline unit tests in `rag-common/tests/test_evals_{concentration,ablation,survey_gold,
judge_calibration}.py` (33 from the original C1 pass + 17 new from the hardening pass:
resolution-sweep-hits-matched-K on synthetic graphs, common-support-scoring drops/keeps refs
correctly, sign-test/bootstrap-CI against hand-computed cases, coverage-stat correctness, and
PROMOTE/TIE/BLOCK verdict rules on constructed paired-comparison inputs) — synthetic
partitions with known concentration, a size-distribution-preservation check on the
size-matched null, and a labeling-sheet round trip. All offline tests in `rag-common` (169
total) pass; `uvx ruff check`/`format` clean. Both tools' own offline suites (GLR 481 passed
/ 15 skipped, RLR 318 passed / 1 known-pre-existing failure / 8 skipped —
`test_query_smoke.py::test_query_self_rag`, unrelated to this change) are unaffected.

## C2 — CLI wiring

Both tools get `eval map-gate [--json] [--profile ...] [--min-refs 100] [--fallback-keywords]
[--target-k K] [--alpha-sweep]` and `eval judge-calibrate [--build-sheet | --score <sheet>]`,
mirrored via `eval/map_gate.py` in each tool (thin adapter over `rag_common.evals` + each
tool's `query/litmap_support.py`/`citation_store.py`/`manifest.py` — same seam as
`map_gate.py`'s sibling `query/litmap_cli.py`).

`map-gate` writes `data/eval/runs/<ts>-map-gate/report.{json,md}` and degrades gracefully
with no citation store at all (text_only-only arm, gate BLOCKs with a named reason) rather
than crashing — verified by running it against RLR before its citation graph existed.

### Provisional run superseded — the granularity-matching bug it exposed

The first `map-gate` pass (2026-07-02, `--n-random` low, `cite enrich` still running) BLOCKed
both tools with `citation_only` outscoring `hybrid` on raw concentration z-scores. Re-running
at `n_random=100` after `cite enrich`/keying settled reproduced the same shape and made the
mechanism visible: GLR's citation graph is only 3.4% both-endpoints-keyed (see keying status
below), so `citation_only` at the default Leiden `resolution=1.0` fragmented into **1,762
clusters on 5,819 docs** (vs. hybrid's ~15) — almost every keyed doc landed in its own
micro-cluster or a same-size pair. A same-partition size-matched null *also* draws from that
tiny-cluster-size distribution, so it doesn't catch the pathology: a survey's few keyed refs
trivially "concentrate" in a ~2-doc cluster purely because there's nowhere else for them to
go, not because the clustering encodes real topical structure. Comparing `citation_only`'s
z-score at 1,762 clusters against `hybrid`'s at 15 clusters was never a valid comparison —
this is the granularity-matching bug the hardening pass below fixes.

### Hardened methodology (`rag_common/evals/ablation.py`, this pass)

1. **Resolution sweep, matched K.** Every arm (`text_only`/`hybrid`/`citation_only`, plus the
   alpha-sweep's interior alphas) is swept over a 15-point Leiden `resolution` grid
   (`0.005`…`8.0`) and the operating point whose **>=2-doc cluster count** (`cluster_count_min2`
   — singletons excluded from the count) is closest to a shared `target_k` is selected.
   `target_k` defaults to the hybrid arm's natural (`alpha=0.5`, `resolution=1.0`) >=2-doc
   cluster count; `--target-k` overrides it. The full score-vs-K curve (every grid point, cheap
   no-null-draw scoring) is in each report's `sweep_curves` for granularity-sensitivity
   auditing.
2. **Coverage, not just cluster count.** `coverage` = fraction of the *whole corpus* landing in
   a >=2-doc cluster; `singleton_share` is the complement. Unkeyed docs under `citation_only`
   are isolated graph nodes — Leiden always makes an isolated node its own singleton cluster
   regardless of resolution — so `citation_only`'s coverage is a direct, resolution-independent
   readout of keying completeness, not a tunable.
3. **Common-support scoring.** Before any arm is scored, survey ref lists are filtered to refs
   placed in a >=2-doc cluster under **every** arm being compared (`restrict_to_common_support`)
   — an arm can no longer get credit for a survey's refs it personally declined to cluster
   while another arm did place them. This is what closes the original pathology: `citation_only`
   no longer gets scored on an easy, self-selected subset of "the refs I happened to key."
4. **Paired comparison, not a bare z-score.** For each common-support survey, `hybrid`'s
   concentration score minus each other arm's score is computed; the promotion gate is decided
   by a percentile bootstrap CI (2,000 resamples) on the mean paired difference plus an exact
   sign test, not by comparing two arms' independent z-scores. **PROMOTE** requires the CI to
   exclude 0 in hybrid's favor against *both* other arms; **TIE** when a CI includes 0; **BLOCK**
   when any arm's CI excludes 0 *against* hybrid.

### Keying status (blocking further OpenAlex enrichment today)

`cite coverage` (both tools, this pass): **GLR 75.9% corpus docs keyed** (4,607/6,071; edges
both-endpoints-keyed 3.4%), **RLR 71.3%** (1,259/1,767; edges both-endpoints-keyed 5.2%). Both
hotmail- and gmail-backed OpenAlex API keys are exhausted for the day; round-robin key rotation
is now in place, and the residual gap (doc-keyed ~75/71% but edge-both-keyed only 3–5%, i.e.
most citation *targets* aren't themselves in-corpus) is blocked on the quota reset at midnight
UTC, not on code. `cite enrich` was **not** re-run in this pass per instructions.

### Final `map-gate` results (2026-07-02, `RAG_EMBED_PROFILE=cuda-qwen0.6b-1024`, `n_random=100`)

Full reports: `graph-layout-rag/data/eval/runs/20260702T174254Z-map-gate/` (default),
`graph-layout-rag/data/eval/runs/20260702T174453Z-map-gate/` (`--alpha-sweep`),
`rag-literature-rag/data/eval/runs/20260702T174519Z-map-gate/` (default),
`rag-literature-rag/data/eval/runs/20260702T174538Z-map-gate/` (`--alpha-sweep`).

**GLR** — doc_count 5,819; 31 qualifying surveys (min_refs=100), 30 common-support; `target_k`=18.

| Arm | alpha | resolution | clusters(>=2) | coverage | z (size-matched) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `citation_only` | 1.00 | 0.75 | 18 | 0.703 | 26.995 |
| `hybrid` | 0.50 | 1.00 | 18 | 1.000 | 24.284 |
| `text_only` | 0.00 | 1.00 | 19 | 1.000 | 18.253 |
| `hybrid_a75` (sweep) | 0.75 | 1.00 | 20 | 1.000 | 19.754 |
| `hybrid_a25` (sweep) | 0.25 | 1.00 | 20 | 1.000 | 14.682 |

Paired (hybrid − arm, 30 common-support surveys): vs `text_only` mean_diff=+0.0022, 95% CI
[-0.0302, 0.0367] (includes 0 → TIE); vs `citation_only` mean_diff=-0.0207, CI [-0.0553,
0.0144] (includes 0 → TIE); vs `hybrid_a75` mean_diff=**-0.0622**, CI **[-0.1015, -0.0200]**
(excludes 0, favors `hybrid_a75`); vs `hybrid_a25` mean_diff=+0.0057, CI [-0.0330, 0.0416]
(includes 0 → TIE).

**GLR verdict: TIE.** Hybrid (alpha=0.5) neither clearly beats nor clearly loses to
`text_only` or `citation_only` once compared at matched granularity and common support — the
BLOCK from the pre-hardening run was the size-mismatched-cluster artifact, not a real result.
The alpha sweep surfaces a real signal the z-only `best_alpha` pick (`hybrid`, by
`mean_z_size_matched`) misses: `hybrid_a75` (alpha=0.75) **significantly beats** plain hybrid
on the paired CI (citation gets more weight in the fusion, consistent with the original
provisional run's intuition that citation signal was underweighted, not spuriously winning).
`z`-score ranking and the paired-CI comparison disagree here — the paired CI is the one the
gate trusts (D10/this pass), so treat `best_alpha`'s z-based pick as a secondary signal only.

**RLR** — doc_count 1,270; 35 qualifying surveys (min_refs=100), 31 common-support; `target_k`=14.

| Arm | alpha | resolution | clusters(>=2) | coverage | z (size-matched) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `citation_only` | 1.00 | 1.00 | 15 | 0.477 | 6.635 |
| `text_only` | 0.00 | 1.00 | 15 | 1.000 | 4.057 |
| `hybrid` | 0.50 | 1.00 | 14 | 1.000 | 3.619 |
| `hybrid_a25` (sweep) | 0.25 | 0.75 | 12 | 1.000 | 2.147 |
| `hybrid_a75` (sweep) | 0.75 | 0.75 | 12 | 1.000 | 1.512 |

Paired (hybrid − arm, 31 common-support surveys): vs `text_only` mean_diff=+0.0280, 95% CI
[-0.0041, 0.0603] (includes 0 → TIE); vs `citation_only` mean_diff=**+0.1347**, CI **[0.0788,
0.1923]** (excludes 0, favors `hybrid`); vs `hybrid_a75` mean_diff=+0.0438, CI [-0.0236,
0.1103] (includes 0 → TIE); vs `hybrid_a25` mean_diff=+0.0174, CI [-0.0294, 0.0601]
(includes 0 → TIE).

**RLR verdict: TIE.** Hybrid clearly beats `citation_only` (CI excludes 0, matching the
intuition that RLR's citation graph is thinner and noisier — only 47.7% coverage even at
matched K, vs. GLR's 70.3% — so pure citation-coupling clustering is the weaker signal here)
but does not clearly beat `text_only`, whose CI includes 0. No alpha variant significantly
beats plain hybrid.

### Read: does hybrid genuinely win, tie, or lose?

Neither corpus produces a genuine BLOCK once granularity is matched and arms are scored on
common support — the original BLOCK verdicts were an artifact of comparing `citation_only` at
~100x hybrid's cluster count. **Both corpora currently gate to TIE**, not PROMOTE: on GLR,
hybrid is statistically indistinguishable from both pure signals at matched K (though
`alpha=0.75` beats plain `alpha=0.5`, worth a default-alpha follow-up); on RLR, hybrid clearly
beats citation-only (the weaker, sparser signal there) but is not distinguishable from
text-only. Reasonable next step before any E2 work: re-run once the citation graphs are more
fully edge-keyed (`cite enrich` resumes after the OpenAlex quota reset), since both tools'
edge-both-keyed rate (3.4%/5.2%) is far below the doc-keyed rate (75.9%/71.3%) — a thin
citation-coupling signal caps how much `citation_only`/`hybrid` can differentiate from
`text_only` regardless of the eval methodology.

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

**Resolved deviation:** the runs above accidentally executed under FP16 fallback (tool venvs
had been re-synced without `--extra retrieval-experiments-gpu`, dropping bitsandbytes). Both
venvs were re-synced with the GPU extra and the floor re-run under a verified clean 4-bit load
(zero fallback warnings in the log). **The 4-bit numbers below are the official regression
floor**; FP16 vs 4-bit shifts every metric by ≤0.007 nDCG@10, i.e. quantization does not move
the floor.

### Official baseline — 4-bit, re-recorded 2026-07-02

| Tool | Track | Strategy | nDCG@10 | MRR | Recall@5 | HR@5 | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GLR | catalog | hybrid | 0.858 | 0.935 | 0.795 | 1.000 | 0 |
| GLR | catalog-nl | hybrid | 0.774 | 0.969 | 0.208 | 1.000 | 0 |
| RLR | catalog | hybrid | 0.539 | 0.423 | 0.857 | 0.881 | 4 |
| RLR | pdf-deep-read | hybrid | 0.531 | 0.414 | 0.854 | 0.878 | 4 |

("Failures" = eval cases with hit_rate@10 < 1.0, a quality stat, not runtime errors.)

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

- **E1 gate run: done, hardened.** See "Final `map-gate` results" above — both tools now gate
  to TIE at matched granularity/common support (default + `--alpha-sweep`, `n_random=100`).
  Not yet PROMOTE; the citation-signal edge-keying gap (3–5% both-endpoints-keyed vs. 71–76%
  doc-keyed) is the most likely lever, blocked on the OpenAlex quota reset (midnight UTC) —
  re-run `eval map-gate --alpha-sweep` on both tools once `cite enrich` resumes and re-keys.
- **Judge calibration labeling sheet:** `eval judge-calibrate --build-sheet` was smoke-tested
  on both tools (cluster-label-sanity slice — lineage/faithfulness calibration needs E1
  lineage output + the user and is explicitly out of scope for this workstream). The
  generated sheets (`data/eval/judge_calibration/cluster_label_sheet.md` in each tool) are
  awaiting the user's hand labels before `--score` can report a real kappa.
- **Profile-footgun fix (map-gate/`map` CLI):** `load_or_build_doc_vectors` (both tools'
  `query/litmap_support.py`) now falls back to the sole other built profile index (with a
  stderr warning) when the *default* profile has no index and exactly one alternative is
  built; an explicit `--profile` that's missing still raises the original clear error, and
  >1 alternative raises a new error listing the built profiles instead of a bare "no index"
  message. RLR's on-disk state has two other built indexes (`cuda-qwen0.6b-1024` and a stale
  `gemini-2-structure-v1` directory left over from the removed Gemini provider), so the
  ambiguous-error path is what actually fires there today, not the silent single-fallback —
  still strictly better than the prior unconditional error with no listed alternatives.
