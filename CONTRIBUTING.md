# Contributing

This repo is a set of five uv-managed Python packages (`rag-common`, `repo-rag`,
`graph-layout-rag`, `rag-literature-rag`, `gpu-gateway`). Lint, format, and tests
are enforced by CI (`.github/workflows/ci.yml`) and a local pre-push hook.

## One-time setup

Install the pre-push hook (runs ruff + secret/large-file guards before every
`git push`, so nothing bad leaves your machine):

```bash
cd tools
uvx pre-commit install --hook-type pre-push
```

The hook lives in `.pre-commit-config.yaml`. It is intentionally fast (lint +
guards only, no tests) — tests run in CI.

## Lint & format

Ruff is the single source of truth (config: `ruff.toml`, shared by all packages).

```bash
cd tools
uvx ruff check .          # lint
uvx ruff check --fix .    # lint + autofix
uvx ruff format .         # format
```

CI runs `ruff check` and `ruff format --check` and **fails on any diff**, so
format before you push (the pre-push hook does this for you).

## Tests

Each package uses pytest. Tests that reach the network, need live API keys, or
need a GPU / real model weights are marked `@pytest.mark.integration` and are
**skipped by default** so the suite runs offline (this is what CI runs):

```bash
cd tools/<package>
uv sync
uv run pytest                       # offline unit subset (integration skipped)
RUN_INTEGRATION=1 uv run pytest     # everything, incl. live/network/GPU
```

`gpu-gateway` is special — its scheduler suite is pure stdlib and needs no sync:

```bash
cd tools/gpu-gateway
uv run --no-project --with pytest --with pytest-asyncio pytest tests/test_scheduler.py
# Live-gateway checks (desktop GPU box only):
uv run --no-project --with pytest --with pytest-asyncio \
  pytest -m integration --gateway-url http://localhost:8765
```

If you add a test that hits the network / an API / the GPU, mark it
`@pytest.mark.integration` — otherwise CI runs it with sockets disabled and it
will fail loudly.

## Lockfiles

Changing dependencies in a package's `pyproject.toml` means re-running `uv lock`
in that package — CI's `lock-check` job fails if `pyproject.toml` and `uv.lock`
drift. (`gpu-gateway` ships no lock; torch is installed out-of-band.)
