# graph-layout-rag — Retrieval Evaluation Findings

Forked from `rag-literature-rag`; same evaluation philosophy (**measurement-bound, not algorithm-bound** — see that fork's `docs/eval-findings.md`). This corpus is graph-drawing / graph-theory literature (Graphviz, Sugiyama layering, stress majorization, orthogonal routing, spectral / community-detection methods) — **jargon-heavy and precise-term**, which makes it behave *differently* from the RAG-literature fork on the lexical-vs-dense axis.

**TL;DR:** on this corpus **lexical retrieval genuinely leads** (bm25 ≈ hybrid > dense) — the *opposite* of the LIT fork — because precise technical terms ("stress majorization", "orthogonal routing", "crossing minimization") are strong exact-match signals. Hybrid is the honest default but barely edges bm25 alone. Every Phase-2 upgrade (learned-sparse, graph-arm, ColBERT) **settled NULL** against hybrid under bootstrap-CI rigor. As in the LIT fork, the durable win was the trustworthy eval, not an algorithm.

---

## 1. Baseline (2026-07-05, de-biased complete-pool qrels)

Local stack: `cuda-qwen0.6b-1024` (Qwen3-Embedding-0.6B, bitsandbytes 4-bit) + Tantivy BM25 + BGE-M3 learned-sparse. Complete 5-system pool (dense/bm25/hybrid/HyDE/multi_query) judged on **DeepSeek** (`deepseek-chat`, ~55k new pairs), scored with `--qrels` overlay, `--fold reporting`:

| GLR nDCG@10   | dense | bm25  | hybrid    | learned-sparse |
| ------------- | ----- | ----- | --------- | -------------- |
| catalog       | 0.740 | 0.799 | **0.822** | 0.684          |
| pdf-deep-read | 0.670 | 0.781 | **0.805** | 0.638          |

Dense went **0.576 → 0.740** (catalog) once scored against the de-biased judged pool instead of gold-only labels — the gold-only metric had been counting dense's real hits as *holes* and punishing it. The `--qrels` overlay is mandatory; omitting it silently scores gold-only (`GRAPH_RAG_QRELS_PATH` is only set by `--qrels`).

## 2. Corpus contrast with the LIT fork (the interesting finding)

| corpus                       | dense vs bm25 |
| ---------------------------- | ------------- |
| **LIT** (RAG/neural-IR lit)  | dense > bm25  |
| **GLR** (graph-drawing)      | **bm25 > dense** |

Same code, **opposite** lexical-vs-dense ordering. Retrieval quality is **corpus-dependent**, driven by how much the query distribution relies on exact technical terms. There is no universal best retriever — pick by the corpus, and re-measure when the corpus changes.

## 3. What did _not_ help (settled — bootstrap 95% CI, do not re-run without new evidence)

- **Learned-sparse (BGE-M3):** loses **significantly** — hybrid−learned Δ catalog **+0.138 [0.087, 0.196]**, pdf **+0.167 [0.110, 0.233]** (n≈48). Root cause was *partly* a real bug: `query/hybrid.py` hard-coded a BM25-tuned `SPARSE_WEIGHT=2.0` and `reciprocal_rank_fusion` applied it to **any** backend, over-amplifying the noisier learned leg below dense. **Fixed** — weights are now env-tunable (`GRAPH_RAG_DENSE_WEIGHT` / `GRAPH_RAG_SPARSE_WEIGHT`, defaults unchanged; 2.0 documented as BM25-specific). At 1:1 the learned arm recovers to ~0.72 but is **still below hybrid 0.82** — BM25 genuinely dominates this jargon corpus.
- **Graph-arm (HippoRAG-2 personalized PageRank + relational router):** the 2026 "graph helps multi-hop" hypothesis **did not hold** — the multi-hop-stratum CI includes 0 (n=25 catalog / 16 pdf, underpowered). graph_arm is a small *general* re-rank lift only (+0.008…0.018, overall CI excludes 0) at **5–6× latency**. No-ship as a routed arm; **registered as an A/B strategy** so it stays benchmarkable. The relational router also misses common cues (e.g. "how do X and Y connect?" routes as non-relational).
- **ColBERT (GTE-ModernColBERT via PyLate/PLAID):** settled NULL at the **environment stage** — pylate pins `sentence-transformers==5.3.0`, incompatible with the `5.5.1` the local dense embedder needs. Blocked on dependency isolation, **not measured to lose**. An existing Qdrant native-multivector MaxSim `colbert` strategy remains for a separate late-interaction test.

## 4. Infrastructure findings

- **Dead default LLM model.** Pool HyDE/multi_query silently ran on **3 of 5 systems** when the default Ollama model (`gemma4:e4b`) wasn't installed (404 warned-and-continued) — a lexically-biased pool. Fixed the run config to a real model + added a **preflight that aborts loudly** if the generation backend is dead. The same dead default silently broke the `research` CLI's decompose step.
- **Pinned deep-learning venvs are fragile.** Out-of-band `torch`/`bitsandbytes` (installed outside the lock) were dropped by a subagent `uv sync` and corrupted by a `uv pip install pylate` (incompatible torch pin). Rule: never install into a tool's `.venv`; isolate heavy DL deps or declare + re-pin them.
- **`--qrels` overlay is load-bearing.** The benchmark scores **gold-only** unless `--qrels <track>/qrels.json` is passed — omitting it discards the entire judged pool and inverts the dense-vs-lexical read.

Full incident / decision log (six incidents caught by verifying artifacts over claims): `docs/research-tool-upgrade-log-2026-07-04.md` (repo root).
