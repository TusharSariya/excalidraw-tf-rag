# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo shape

`excalidraw-tf-rag` is a workspace of four independent, uv-managed Python 3.11+ packages (no root `pyproject.toml` — `ruff.toml` is the single lint/format source of truth):

- **`rag-common`** — shared local embedding provider (`local` backend only: sentence-transformers / bitsandbytes-4bit CUDA / MPS), named embed profiles, the pluggable LLM-generation transport (`rag_common.local_llm`: Ollama default, plus opencode and GPU-gateway backends), reranking, and the S3 backup/restore tooling (`scripts/`, `backup-manifest.toml`).
- **`graph-layout-rag`** — local research corpus for graph-drawing/layout theory (Graphviz, Sugiyama layering, stress majorization). Used by agents researching Terraform pipeline layout height.
- **`rag-literature-rag`** — local research corpus for core RAG/neural-IR literature (DPR, ColBERT, GraphRAG, agentic RAG). Forked from `graph-layout-rag` with domain-specific harvest/relevance/taxonomy.
- **`gpu-gateway`** — a persistent Ray Serve inference gateway that serves embedding/reranking/LLM models within one VRAM budget on a single consumer GPU (RTX 3060 Ti / 8 GB target), with bin-packing co-residency and atomic LRU eviction. OpenAI-compatible HTTP surface.

`graph-layout-rag` and `rag-literature-rag` share near-identical internal structure (`ingest/`, `eval/`, `query/`, `paths.py`, `env.py`) — when fixing something in one, check whether the same fix applies to the other.

All embedding is **local-only** (no OpenAI, Gemini, or MLX providers — those were removed; do not reintroduce them). LLM text-generation (HyDE/query-transform, contextual retrieval, eval judge) defaults to a local Ollama server, with opencode and the GPU-gateway HTTP endpoint as alternate backends.

## Navigating this repo

This repo has a prebuilt `graphify` knowledge graph at `graphify-out/graph.json`. Per `.cursor/rules/graphify.mdc` (mirrored for Claude Code as the `.agents/skills/graphify` skill, symlinked into `.claude/skills/`): **before using Read/Grep/Glob/Bash to explore code or architecture, run graphify first** — `graphify query "<question>"`, `graphify path "<A>" "<B>"`, `graphify explain "<concept>"`. This applies to every subagent too. Only fall back to Read/Grep/Glob directly once graphify has oriented you, or if `graphify-out/graph.json` doesn't exist. The graph was last rebuilt before the repo-rag deletion and Gemini/OpenAI/MLX removal — treat any node referencing those as stale until `graphify` is re-run.

## Commands

Install the pre-push hook once (ruff + secret/large-file guards before every `git push`):

```bash
cd tools && uvx pre-commit install --hook-type pre-push
```

Lint & format (ruff config is shared across all packages via root `ruff.toml`):

```bash
uvx ruff check .          # lint
uvx ruff check --fix .    # lint + autofix
uvx ruff format .         # format
```

Tests, per package (integration tests — network/API keys/GPU — are marked `@pytest.mark.integration` and skipped by default so the suite runs offline, matching CI):

```bash
cd tools/<package>
uv sync
uv run pytest                       # offline unit subset
RUN_INTEGRATION=1 uv run pytest     # everything, incl. live/network/GPU
```

`gpu-gateway` is special — its scheduler suite is pure stdlib, no sync needed:

```bash
cd tools/gpu-gateway
uv run --no-project --with pytest --with pytest-asyncio pytest tests/test_scheduler.py
```

Changing a package's dependencies requires re-running `uv lock` in that package — CI's `lock-check` job fails on `pyproject.toml`/`uv.lock` drift (`gpu-gateway` ships no lock; torch is installed out-of-band).

## Versioned packages & changelogs

Some packages publish a Docker image via the `build-push` CI workflow, tagged with the `version` in their `pyproject.toml` (e.g. `v3/harvest`). The workflow's version guard **rejects re-publishing a tag that already exists in prod ECR** — prod publishes on merge to `master`, so a merged version is permanently taken.

**When you change a versioned package and push, keep three things in sync:**
1. **Bump `version` in its `pyproject.toml`** (and re-run `uv lock`).
2. **Add a `CHANGELOG.md` entry** under that version.
3. **Update its `README.md`** if the architecture, configuration, or usage changed.

(Docs-only edits that don't ship in the image don't need a bump. A package may carry its own `CLAUDE.md` with additional, package-specific rules.)

## Embedding profiles

Both RAG tools resolve embeddings through named profiles in `rag-common/src/rag_common/embed_profiles.toml` (built-ins) merged with a tool-local `embed_profiles.toml` override (see `ingest/embed.py` in each tool). List them with `uv run <tool> embed profiles`. Production profiles are the `cuda-*` (CUDA/bitsandbytes 4-bit) and `local-fp16-*`/`qwen3-code` (plain sentence-transformers) profiles — pick a profile by the quant/device it names, not by assuming a specific backend.

## GPU gateway integration

`RAG_GPU_GATEWAY_URL` (if set) redirects local-backend embed calls in `rag_common.embed` through the gateway's `/v1/embeddings`-style surface instead of loading a model in-process — used when a tool runs on a machine other than the one hosting the GPU. `rag_common.local_llm`'s `gateway` backend does the same for text generation.
