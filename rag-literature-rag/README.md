# RAG literature RAG

Local research corpus for **core retrieval-augmented generation** literature: foundations (DPR, ColBERT, Lewis RAG), chunking, hybrid retrieval, GraphRAG, agentic RAG, and evaluation benchmarks. Forked from [graph-layout-rag](../graph-layout-rag) with domain-specific harvest, relevance, and taxonomy.

## Setup

```bash
cd tools/rag-literature-rag
uv sync
cp .env.example .env
```

Embeddings use shared [`tools/rag-common`](../rag-common) with named profiles (`RAG_EMBED_PROFILE` or `--embed-profile`).

| Profile | When |
| --- | --- |
| `cuda-qwen0.6b-1024` | **Production query** |
| `cuda-qwen0.6b-section-v1` | Chunking campaign: enrich indexed text with document/section metadata |
| `cuda-qwen0.6b-small2big-v1` | Chunking campaign: smaller child chunks (~350 target / 550 max) |
| `cuda-qwen0.6b-small2big-dual-v1` | Parent-child A/B: retrieve small child chunks, return baseline-sized parent passages |
| `cuda-qwen0.6b-contextual-v1` | Contextual Retrieval profile (local LLM-generated context lines are cached in `data/indexes/contextual_cache.json`; the index itself has not been built yet — run `ingest --embed-profile cuda-qwen0.6b-contextual-v1 --force --rebuild` to build it) |
| `cuda-qwen0.6b-longrag-v1` | LongRAG/larger-unit A/B: larger section/page-span chunks (~1800 target / 2400 max) |
| `cuda-qwen0.6b-512-v1` / `cuda-qwen0.6b-768-v1` | Qwen3-0.6B reduced-dimension smoke profiles; not promoted |
| `cuda-qwen4b-1024` / `cuda-qwen4b-2560` | Qwen3-4B CUDA quality probes; skipped on RTX 3060 Ti 8 GB after OOM |
| `mlx-qwen4b-raptor-gemma4-v1` | Local RAPTOR experiment: MLX embed + Ollama recursive tree summaries (opt-in) |
| `cuda-qwen0.6b-raptor-gemma4-v1` | Desktop RAPTOR experiment: CUDA Qwen3-0.6B embed + Ollama Gemma4 tree summaries (opt-in) |

## Local-first embedding

Build/rebuild the production index directly on the GPU box, query locally:

```bash
RAG_EMBED_PROFILE=cuda-qwen0.6b-1024 uv run rag-literature-rag ingest --force --rebuild
yarn rag-lit:query "Self-RAG reflection tokens" --top 8 --json
```

## Chunking and contextual experiments

The 2026-06-18 quality campaign keeps the production default unchanged and adds local-first A/B profiles:

- `cuda-qwen0.6b-section-v1`: default chunk boundaries, but indexed dense/BM25 text is prefixed with document title, year, authors, source URL, section path, categories, tags, and aliases.
- `cuda-qwen0.6b-small2big-v1`: smaller child chunks (`target=350`, `max=550`, `overlap=80`) with a distinct chunking fingerprint and chunk-cache key.
- `cuda-qwen0.6b-small2big-dual-v1`: true small-to-big retrieval. Ingest builds child chunks (`target=350`, `max=550`, `min=120`, `overlap=80`) and parent chunks (`target=800`, `max=1200`, `min=200`, `overlap=120`) from the same structural blocks. Dense retrieval searches child rows, aggregates to parent rows, and `--small-to-big` returns parent evidence with child-hit metadata. Parent BM25 lives under `bm25_parent/`.
- `cuda-qwen0.6b-contextual-v1`: default chunking plus local-LLM contextual prefixes through `rag_common.local_llm` (`RAG_LLM_BACKEND=ollama` on `desktop`). Status: the generated context lines are cached in `data/indexes/contextual_cache.json` (~20k entries) so a build can reuse them cheaply, but the profile's LanceDB/BM25 index has not actually been built — `data/indexes/` only has `cuda-qwen0.6b-1024` on disk. Build pending; do not assume this profile is queryable yet.
- `cuda-qwen0.6b-longrag-v1`: larger retrieval units (`target=1800`, `max=2400`, `min=600`, `overlap=240`) for a non-generation LongRAG probe.
- `cuda-qwen0.6b-512-v1` / `cuda-qwen0.6b-768-v1`: reduced-dimension Qwen3-0.6B profiles. The 50-document smokes completed cleanly, but full-corpus benchmarks are required before recommendation.

Extraction caching has three reusable layers. Profile-specific chunk caches make reruns of the same profile fast. Backend page caches are keyed by PDF SHA, extraction backend, and extraction options, so future chunking profiles can reuse expensive Docling page extraction and only redo chunking plus embedding. Embedding cache entries are keyed by backend, model, dims, quant, title, and final indexed text, not profile name.

Build each as its own index and benchmark against synthetic qrels:

```bash
RAG_EMBED_PROFILE=cuda-qwen0.6b-section-v1 uv run rag-literature-rag ingest --force --rebuild -v
uv run rag-literature-rag eval benchmark --embed-profile cuda-qwen0.6b-section-v1 \
  --qrels data/eval/qrels/catalog/qrels.json --strategy dense --strategy bm25 --strategy hybrid --report -v
```

Parent-child A/B benchmark:

```bash
RAG_EMBED_PROFILE=cuda-qwen0.6b-small2big-dual-v1 uv run rag-literature-rag ingest --force --rebuild -v
uv run rag-literature-rag query "reciprocal rank fusion parent chunks" \
  --embed-profile cuda-qwen0.6b-small2big-dual-v1 --small-to-big --json
uv run rag-literature-rag eval benchmark --embed-profile cuda-qwen0.6b-small2big-dual-v1 \
  --qrels data/eval/qrels/catalog/qrels.json \
  --strategy dense --strategy bm25 --strategy hybrid \
  --strategy small2big_dense --strategy small2big_parent_bm25 --strategy small2big_hybrid --report -v
```

Campaign report: [docs/quality-campaign-2026-06-18.md](docs/quality-campaign-2026-06-18.md).

Current result: no campaign profile is promoted over `cuda-qwen0.6b-1024`; production stays on default `hybrid`. Section improves some nDCG scores but bpref falls and `hole@10` remains nonzero. Small2big completes cleanly at 47,219 chunks but regresses catalog/fusion quality. The dual profile builds parent evidence and improves raw small2big hybrid nDCG, but benchmark failures and diagnostic holes remain nonzero. Docsummary, contextual, LongRAG, and reduced-dimension profiles are smoke-tested only.

Next campaign order:

1. Run corpus health first; the audit reports fallback causes and qrel-heavy fallback documents.
2. Full-corpus `cuda-qwen0.6b-longrag-v1`, benchmarked on `dense`, `bm25`, and `hybrid` for catalog and PDF tracks.
3. Full-corpus `cuda-qwen0.6b-768-v1`, then `cuda-qwen0.6b-512-v1`, using the same benchmark/diagnostic gate plus index size, RSS, and build time.
4. Full `cuda-qwen0.6b-docsummary-gemma4-v1` only after the cheap full-corpus candidates pass.

Promotion requires nDCG and bpref agreement, `hole@10 = 0`, zero failures, and acceptable latency/memory. Reduced-dimension wins should be documented as cost profiles unless they also clearly improve quality.

## Pipeline

```bash
# From repo root
yarn rag-lit:harvest -- --deep-harvest --target-pdfs 1000 --workers 32 --resume -v
yarn rag-lit:harvest verify
yarn rag-lit:harvest enrich
yarn rag-lit:ingest -- --force --rebuild
yarn rag-lit:query "GraphRAG community summary" --top 8 --json
yarn rag-lit:catalog
```

## Deep research (`research`)

`research <question>` is the agentic synthesis arm on top of plain `query`: it decomposes
the question into ≤4 sub-questions, runs hybrid retrieval per sub-question, follows the
citation graph 1 hop from the top docs (keeping neighbors that clear an embed-similarity
bar — no LLM), summarizes each unique doc, then writes a cited synthesis with inline
`[bib_key]` citations, a references section, and a `SUFFICIENT: yes/no` self-report that
gates at most one iterate round.

```bash
uv run rag-literature-rag research "What did contextual retrieval change about chunk augmentation?" \
    --max-docs 6 --embed-profile cuda-qwen0.6b-1024      # human-readable markdown
uv run rag-literature-rag research "..." --json          # structured agent surface
uv run rag-literature-rag research "..." --dry-run        # retrieval + cite-expansion only, ZERO LLM calls
```

Flags: `--top` (retrieval depth/sub-question), `--max-docs` (≤12, caps evidence LLM calls),
`--depth` (`1` single synthesis, `2` allow one iterate), `--dry-run`, plus the usual
`query` filters (`--tag/--category/--source/--year-min/--pdf-only`). The end-of-run line
prints the running LLM call count + a rough token/USD estimate.

**LLM backends** (two lanes, all env-overridable). Cheap steps (decompose, follow-up) route
through `rag_common.local_llm` by default (`RAG_LIT_RESEARCH_CHEAP_BACKEND`, default `local`);
synthesis + evidence summaries default to a direct-HTTP DeepSeek helper
(`RAG_LIT_RESEARCH_SYNTH_BACKEND`, default `deepseek`, `deepseek-chat` via `DEEPSEEK_API_KEY`).
Set either lane to `deepseek`/`opencode`/`local`. On a box with no local Ollama models, run the
cheap lane on DeepSeek too: `RAG_LIT_RESEARCH_CHEAP_BACKEND=deepseek`. A typical `--max-docs 6`
run is ~8–14 calls (well under a cent on DeepSeek). If `query/graph_arm.py` is present it is used
as the retrieval strategy automatically (soft optional; `--no-graph-arm` to force plain hybrid).

### Harvest sources

- **Topic seeds** (~50 canonical papers): Lewis RAG, DPR, ColBERT, Self-RAG, CRAG, RAPTOR, GraphRAG, agentic surveys, RAGAS, etc.
- [awesome-generative-ai-guide RAG table](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md) (core tags only)
- OpenAlex topic queries (foundations, GraphRAG, agentic, evaluation, …)
- arXiv (`cs.CL`, `cs.IR`, `cs.AI`)
- Crossref venues (ACL, EMNLP, SIGIR, WWW, NeurIPS) — strict RAG relevance gate
- Bibliography chain + forward citations from seed DOIs

Target corpus: **~800–1200 open-access PDFs** (core RAG only; no domain-specific medical/legal RAG).

### Query tags / categories

`foundations`, `dense-retrieval`, `hybrid-retrieval`, `chunking`, `query-expansion`, `reranking`, `self-correcting`, `graphrag`, `agentic`, `memory`, `long-context`, `evaluation`, `training`, `engineering`, `survey`

Default retrieval: **hybrid** (Gemini dense + Tantivy BM25 + RRF k=20).

### Eval

```bash
uv run rag-literature-rag eval corpus-health --embed-profile cuda-qwen0.6b-1024 --track catalog --json  # read-only audit, no LLM
uv run rag-literature-rag eval corpus-health --embed-profile cuda-qwen0.6b-1024 --track pdf-deep-read --json
uv run rag-literature-rag eval validate-gold --json
yarn rag-lit:eval -- --embed-profile gemini-2-structure-v1 --report -v
```

Gold set: 42 curated agent-realistic queries in `src/rag_literature_rag/eval/gold_cases.py`. Build neutral qrels with multi-system pooling (`eval pool` → `eval judge`) before comparing strategies, and judge wins on `eval diagnostics` (hole_rate@k / bpref), never raw nDCG (pooling bias).

**Synthetic gold expansion** (`eval gen-gold`): 42 cases can't _distinguish_ retrieval methods (they cluster in a ~0.11 nDCG band). Generate a large stratified synthetic gold set to make the eval discriminate — system-blind Flash queries grounded in corpus docs, anti-leakage + de-dup filtered, then the existing pool→judge pipeline. Opt-in via `RAG_LIT_SYNTH_GOLD=1`; curated-42 stays the default.

```bash
uv run rag-literature-rag eval gen-gold --embed-profile gemini-2-structure-v1 \
  --n-catalog 300 --n-pdf 200 --hard-frac 0.2 --budget-usd 12 -v
uv run python scripts/synth_separation_report.py   # curated-vs-synthetic spread + Spearman
```

On 531 cases the method spread widens **3.5–4×** (methods separate), agrees with the human anchor (Spearman 0.83–0.94), and is pooling-unbiased (hole@10 = 0). Full writeup: [docs/eval-findings.md](docs/eval-findings.md).

## Agent skill

[`.agents/skills/rag-literature-rag/SKILL.md`](../../.agents/skills/rag-literature-rag/SKILL.md)
