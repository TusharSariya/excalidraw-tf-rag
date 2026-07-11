# DB tests

## `e2e_fanout.py` — fan-out end-to-end

Proves a seed fans out to every harvest queue, end-to-end, against the **real**
objects — `public.seeds`, `public.publish_seed`, and the `harvest_arxiv` /
`harvest_openalex` / `harvest_s2` queues.

It's faithful to how the real system works: a single **direct Postgres**
connection does the write (the way the seeder pushes — connecting directly and
calling the `SECURITY INVOKER` `publish_seed`, never PostgREST), the verify, and
the purge.

It's a self-contained [PEP 723](https://peps.python.org/pep-0723/) script — `uv`
resolves its one dep (`psycopg`) on the fly; no `uv sync` needed.

> [!WARNING]
> **This test is destructive.** It purges the real `harvest_*` queues and
> `truncate`s `public.seeds`. Run it **only against a disposable, ephemeral
> database** — the local `supabase start` stack (below), or a throwaway preview
> branch. **Never** point it at staging or prod.
>
> As a guard, the script **refuses to run** unless `SUPABASE_DB_URL` points at a
> loopback host (`127.0.0.1` / `localhost` / `::1`). For a deliberately-disposable
> remote DB (e.g. a preview branch) set `E2E_ALLOW_NONLOCAL=1` to override.

### Run against a local ephemeral stack

```bash
supabase start --workdir v3
supabase db reset --workdir v3          # replays all migrations onto a fresh Postgres

SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \
uv run v3/supabase/tests/e2e_fanout.py
```

Expected: each of the three `harvest_*` queues shows depth 1 with the published
`paper_id`, then the finale purge empties them.

### CI

The `validate` job in `.github/workflows/db-migrations.yml` runs this on every PR
that touches `v3/supabase/migrations/`. It spins up an ephemeral local stack
(`supabase start` + `supabase db reset`) — a fresh, isolated Postgres built from
the same migrations and destroyed when the job ends — and runs the script against
it (`SUPABASE_DB_URL` pointing at `127.0.0.1:54322`). No secrets required.
