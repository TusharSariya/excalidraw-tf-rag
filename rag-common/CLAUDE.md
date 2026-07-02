# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rag-common` has no README — this is the only doc for it. It's the shared library `graph-layout-rag` and `rag-literature-rag` both depend on (`path = "../rag-common", editable = true` in their `pyproject.toml`), so a change here affects both tools; run both tools' test suites after editing.

## Module map

- **`embed.py`** — `embed_texts`/`embed_query`, the entry point every tool calls. Routes to `remote_embed.py` when `RAG_GPU_GATEWAY_URL` is set, otherwise `local_embed.py`. `resolve_workers` (generic concurrency sizing) lives here too.
- **`local_embed.py`** — the only embed backend: sentence-transformers, with a bitsandbytes-4bit CUDA path (`use_cuda_bnb_4bit` in `config.py`) and plain CPU/MPS fallback.
- **`config.py`** — `EmbedConfig`/`EmbedBackend` (`Literal["local"]` — this is the only backend; do not add cloud backends back), `local_config()`, `resolved_embed_backend()` (host-specific quant tag like `cuda-bnb-4bit` vs `local-4bit`, used by each tool's HG1 backend-match guard to stop a `cuda-*` index being queried on a mismatched quant stack).
- **`profiles.py`** + **`embed_profiles.toml`** — named embed profiles (TOML), merged with each tool's local `embed_profiles.toml` override. `resolve_embed_profile_config` also supports a legacy `RAG_EMBED_BACKEND=auto|local` env-var path (the implicit `default` profile).
- **`local_llm.py`** — pluggable text-generation transport used by HyDE/query-transform, contextual retrieval, and the eval judge across both tools. `RAG_LLM_BACKEND`: `ollama` (default, local server), `opencode` (CLI, e.g. `opencode-go/deepseek-v4-flash`), `gateway` (routes through the GPU gateway's `/v1/chat/completions`).
- **`rerank.py`** — local cross-encoder reranking (`BAAI/bge-reranker-v2-m3` default) plus LLM listwise rerank via `local_llm`.
- **`remote_embed.py`** — client for the `gpu-gateway` embedding endpoint.
- **`scripts/`** + **`backup-manifest.toml`** — S3 backup/restore/verify for every tool's `data/` (declarative manifest; a new tool is one `[tool.<name>]` block, a new embed profile needs no edit since categories are globs).

## Testing

```bash
cd rag-common
uv run --with pytest-socket pytest -m "not integration" --disable-socket --allow-unix-socket -q
```

`tests/conftest.py` maintains an explicit `_INTEGRATION` node-id set (network/live-Ollama tests) rather than relying only on markers scattered in test files — add to it if you add a test that needs the network or a live server.
