# Decision Log — Research-Tool Campaign (2026-07-02)

Chronological record of every decision behind the plan in
`~/.claude/plans/take-a-look-at-resilient-naur.md`. Format: **decision — rationale**.

## Phase 1 — Framing (user clarifications, pre-plan)

1. **Target metric: both, retrieval-first** — retrieval qrels as fast inner loop, LLM-judged
   end-to-end validation for winners.
2. **Ingest budget: anything, incl. overnight unattended runs** on the 3060 Ti.
3. **Scope: both tools equally** — shared implementation in rag-common, thin per-tool wiring.
4. **Research depth: full deep-research harness** (fan-out search + adversarial 3-vote
   verification), not a light search pass.
5. **REFRAME (user, mid-session): not "a better RAG" but "the best research tool"** for humans
   and agents to understand RAG literature / graph-layout theory — broader than retrieval.
6. **Capabilities: all four** — landscape mapping, deep synthesis Q&A, claim/method extraction,
   agentic deep-research.
7. **Consumers: humans and agents equally** — same engine, two output surfaces.
8. **Eval philosophy: build task-level evals** (retrieval qrels stay as regression floor; new gold
   sets + LLM-judge protocols for the new jobs) — keeps the no-BS promotion-gate discipline.
9. **Ambition: extend the existing stack** (LanceDB + Tantivy + citations.sqlite +
   rag_common.local_llm) — no new knowledge-graph platform, no rewrite.

## Phase 2 — Evidence rulings (deep-research, adversarially verified)

10. **PageRank-for-ranking: dead** (user's idea; already tested NULL in-repo even at 89% keying) —
    but the PPR/coupling/co-citation machinery is repurposed for mapping as anchor discovery.
11. **Entity-level GraphRAG: cut** — advantage confined to LLM-judged "global sensemaking"
    (72–83% win rates, position-bias caveats, 2404.16130); loses to plain RAG on SQuALITY/QMSum
    ROUGE and single-hop QA under a unified protocol (2502.11371). Only cluster/community
    summaries survive, deferred to E2 (C0 summaries cut query tokens ~97%).
12. **RAPTOR: stays dead** — local NULL + literature shows headline gains need GPT-4-class
    readers (retrieval-only +1.8–4.5 F1 on QASPER).
13. **Landscape mapping = the evidence-backed build**: Connected Papers is co-citation +
    bibliographic coupling only (already implemented here); hybrid coupling+text clustering beats
    pure citation on all measures (Boyack & Klavans, 2.15M docs); main-path analysis is the
    standard lineage method; reference-concentration on ≥100-ref surveys is a labels-free eval.
14. **Small local LLMs don't drive agentic loops well** → E4 inverts: expose corpus as JSON/MCP
    tools so the caller's strong model drives; local model only summarizes.

## Phase 3 — Eng review decisions (D1–D11, plan-eng-review + codex outside voice)

15. **D3 scope: commit E0+E1(+its eval) only; E2–E4 = designed follow-ups, each go/no-go with
    prior gate data** — matches existing campaign discipline; avoids building on unproven layers.
16. **D4 doc vectors: mean-pooled chunk embeddings from the champion index** (chunk-count-
    weighted, profile/version-keyed) — 100% coverage both tools, zero new compute. SPECTER2 =
    optional GLR-side ablation only (82% coverage, 0% on RLR, tested negative for ranking).
17. **D5: citation-graph coverage precheck + repair in E0** — kills the false-NULL mechanism
    (echo of the silent-BM25 bug: verify infrastructure before measuring algorithms).
18. **D6: new eval code lives in rag-common `evals/`**, per-tool code = thin adapters — the two
    tools' eval/ folders are documented-drifting copy-forks; stop feeding that.
19. **D7: judge calibration before any judge-gated metric** — user hand-labels ~30 cases against
    a written rubric; deepseek-v4-flash must hit κ ≥ 0.7; fail path: revise prompt → re-test once
    → else gate stays human-judged. Motivated by the 29%-poisoned-qrels judge bug.
20. **D8 (codex tension, accepted): E5 split** — first slice builds only map-eval + judge
    calibration; synthesis gold/ALCE faithfulness/position-swap judging move into E2.
21. **D9 (codex tension, accepted): tree reconciliation = hard blocker** — later resolved by user
    (see #27).
22. **D10 (codex hardening batch, all 12 accepted):** cycle handling + temporal validation before
    main-path SPC; cluster-size-aware concentration metric with size-matched nulls + per-section
    survey gold; rank-based/kNN-sparsified fusion (coupling and cosine aren't commensurate);
    doc-vector versioning; extractive TF-IDF labels default, LLM labels opt-in; `map seminal` =
    anchor discovery, never ranking; JSON-first output contract; pin exact regression command;
    E2+E3 share one per-doc schema; judge-calibration fail path defined; drop 4B-contextual note
    from plan; flag igraph/leidenalg native deps.
23. **D11: `cuda-qwen4b-contextual-v1` benchmark → TODOS.md** — last open retrieval question,
    parked out of this campaign.
24. Eng review verdict: CLEAR (scope-reduced), 4 issues, 0 critical gaps, no codex point rejected.

## Phase 4 — Execution-start decisions (post-review)

25. **Canonical tree = THIS repo** (user, 2026-07-02) — the "desktop tree is canonical" note in
    docs/rag-gap-phase0-execution-decisions.md is stale; do not block on it.
26. **Workspace committed** (`1ff0c20`) — graphify graph, agent skills, TODOS.md, gpu-gateway
    lock; work branch `research-tool-e0-e1`.
27. **Audit findings (2026-07-02) upgraded E0.2 from "port repair" to "build + implement":**
    - GLR citations.sqlite is in pre-repair state here: 1,334/5,811 corpus docs keyed (~23%),
      9.1% of 100,362 edges have both endpoints keyed.
    - RLR has **no citations.sqlite at all**; `cite enrich` CLI exists but was never run.
    - The 2.5%→89% keying repair from the gap-analysis ledger is **not in this tree**; keying is
      DOI-only. New work: arXiv-ID + normalized-title matching, target ≥80% keyed docs per tool,
      plus a re-runnable `cite coverage` audit subcommand.
28. **RLR regression floor pinned to `cuda-qwen0.6b-1024`** — the default `contextual-v1` profile
    has no index on this machine; floor uses the index that exists (noted in script header).
29. **Execution style (user): sonnet subagents run the plan** to keep the orchestrator's context
    clean; sequential dispatch (shared files: cli.py, rag-common/pyproject.toml), citation
    enrichment runs in background throughout.

## Phase 5 — Execution decisions (2026-07-02, during the run)

30. **Chain restructure for speed (user asked "can it be done faster?")** — killed the 45-min
    OpenAlex cooldown, ran re-keys at `--workers 4` (~4 rps, under the 10 rps limit) and
    chained the final map-gates directly on. Verdict on the gamble: GLR's resolution pass ran
    clean until the *key's own daily quota* died — throttling was never about the cooldown.
31. **OpenAlex key rotation = round-robin, not just failover** (user request). GLR's
    first-available selection replaced with round-robin over non-depleted keys; RLR ported
    from single-key to the full multi-key policy. `OPENALEX_API_KEYS` comma list in each
    gitignored `.env`; both user keys (hotmail + gmail) configured. Commit `dc457cc`.
32. **Keying stopped at 75.9% (GLR) / 71.3% (RLR), not the 80% target** — both keys' OpenAlex
    daily quotas exhausted (429 storms with 0 successes), plus the client-side $1/run spend
    cap drains on retries. Decision: accept today's coverage, re-run after midnight UTC.
    The doc-level residual matters less than the **edge-keying gap** (3.4%/5.2%
    both-endpoints-keyed), which caps the citation-coupling signal itself.
33. **Map-gate profile footgun fixed** — RLR's gate initially crashed because `.env` defaults
    to index-less `contextual-v1`. Now: silent-with-warning fallback when exactly one other
    index exists, explicit listing error when several do.
34. **Gate methodology superseded (the big one).** The first final verdicts (BLOCK on both
    tools, citation-only "winning") were a granularity artifact — citation-only fragmented
    into 1,762 micro-clusters on 5.8k docs and micro-clusters game concentration even against
    size-matched nulls (exactly the codex-review warning, in a form the D10 hardening hadn't
    fully covered). Fix: per-arm Leiden resolution sweep to a matched target K, coverage /
    singleton-share diagnostics, common-support scoring, paired bootstrap-CI + sign-test
    verdicts (PROMOTE/TIE/BLOCK). Commit `bb2dccd`. **Hardened verdicts: TIE on both tools** —
    the earlier BLOCKs were measurement, not reality.
35. **Live signal worth chasing:** GLR α=0.75 (citation-leaning fusion) beats α=0.5 with
    paired CI excluding 0; RLR hybrid clearly beats citation-only but ties text-only.
    Interpretation: citation signal is real but starved by edge keying (see 32).
36. **E1 disposition:** the `map` command family ships regardless of the gate — the gate
    governs whether *hybrid clustering* becomes the default map substrate, not whether the
    tools exist. E2's substrate defaults to text-only clusters if the post-rekey gate still
    ties.
37. **Push blocked at session end** — user's new fine-grained PAT is personal-scoped and
    cannot see `TusharSariyaOrg/excalidraw-tf-rag`; needs org resource-owner or classic
    `repo`-scope token. PR deferred until then.
