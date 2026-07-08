# Contributing

This repo is a set of four uv-managed Python packages (`rag-common`,
`graph-layout-rag`, `rag-literature-rag`, `gpu-gateway`). Lint, format, and tests
are enforced by CI (`.github/workflows/ci.yml`) and a local pre-push hook.

## One-time setup

Install the pre-push hook (runs ruff + secret/large-file guards before every
`git push`, so nothing bad leaves your machine):

```bash
cd tools
uvx pre-commit install --hook-type pre-push
```

The hook lives in `.pre-commit-config.yaml`. It's intentionally fast for
everyday pushes (lint + guards only, no tests — tests run in CI), with one
exception: pushing a change to `.github/workflows/**` also runs `actionlint`
(static, still fast) and dry-runs the changed workflow(s) with `act` (real
Docker execution — slower, only fires on workflow changes). See "Testing GH
Actions workflows locally" below.

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

## Testing GH Actions workflows locally

Two tools, for two different kinds of bug — use both, they don't overlap:

- **`actionlint`** — static validation of workflow YAML: schema, expression
  syntax, known action inputs, `shellcheck` on `run:` blocks. Catches things
  like referencing `secrets` directly inside an `if:` conditional (invalid —
  breaks the *entire* workflow file, not just that step). Runs automatically
  as part of the pre-push hook; no install needed, pre-commit manages the
  binary itself.

- **`act`** — actually executes a workflow locally in Docker, simulating a
  real `pull_request`/`push` event. Catches logic bugs `actionlint` can't see
  (wrong CLI flags, a step that silently no-ops instead of erroring). Also
  wired into the pre-push hook, but scoped to only run when a workflow file
  is part of what you're pushing — everyday code pushes stay fast.

  One-time install: `brew install act`. The hook script
  (`scripts/run-act-checks.sh`) auto-detects your Docker socket (OrbStack,
  Colima, Docker Desktop, ...) and pins a non-interactive runner image, so it
  should just work once `act` is installed.

  To dry-run a specific workflow by hand (e.g. while iterating on it, before
  even committing):

  ```bash
  act pull_request -W .github/workflows/ci.yml \
    -e .github/act-event-pull_request.json \
    -P ubuntu-latest=catthehacker/ubuntu:act-latest
  ```

  `.github/act-event-pull_request.json` is a minimal fake PR event — several
  workflows read `github.event.pull_request.base.sha` directly, and without a
  payload `act` passes an empty event, which defeats the point.

  **Known flakiness, not necessarily a real bug:**
  - `db-migrations.yml`'s `validate` job runs `supabase start`
    (Docker-in-Docker) — nested Docker under `act` can be slow/flaky on its
    own.
  - Matrix jobs in `ci.yml` run as concurrent local containers sharing the
    same cache directories (e.g. `setup-uv`'s cache) — unlike GitHub's
    isolated hosted runners, this can race and fail a job that's actually
    fine.

  If a job fails under `act` but you're not sure it's real, re-run just that
  job in isolation with `-j <job-name>` before assuming the workflow itself
  is broken.
