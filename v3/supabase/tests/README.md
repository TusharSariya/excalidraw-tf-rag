# DB tests

## `e2e_fanout.py` — fan-out end-to-end

Proves a seed fans out to every harvest queue, end-to-end, against the
`_test` mirror surface (`seeds_test`, `publish_seed_test`, `harvest_*_test`
queues — created by the `create_test_surface` migration). It never touches real
data or the real `harvest_*` queues, and purges everything it created at the end.

Faithful to how the real system works: the **write** goes through the Supabase
client (the way the seeder pushes); **verify + purge** go over a direct Postgres
connection (the way the KEDA scaler / worker read pgmq — pgmq is never in the
REST path).

It's a self-contained [PEP 723](https://peps.python.org/pep-0723/) script — `uv`
resolves its deps (`supabase`, `psycopg`) on the fly; no `uv sync` needed.

### Run against a local ephemeral stack (no staging pollution)

```bash
supabase start --workdir v3
supabase db reset --workdir v3          # replays all migrations, incl. the _test surface

# service_role key + DB URL come from `supabase status`
SUPABASE_URL=http://127.0.0.1:54321 \
SUPABASE_KEY="$(supabase status --workdir v3 -o env | sed -n 's/^SERVICE_ROLE_KEY=//p' | tr -d '"')" \
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \
uv run v3/supabase/tests/e2e_fanout.py
```

Expected: each of the three `harvest_*_test` queues shows depth 1 with the
published `paper_id`, then the finale purge empties them.

### Run against staging

Same command, pointing at staging:

- `SUPABASE_URL=https://vumvzcugxbnkfnujpjjc.supabase.co`
- `SUPABASE_KEY=<staging service_role key>`
- `SUPABASE_DB_URL=postgresql://postgres.vumvzcugxbnkfnujpjjc:<DB password>@aws-0-us-east-2.pooler.supabase.com:5432/postgres`
  (session pooler — IPv4; direct `db.<ref>.supabase.co` is IPv6-only)

### CI

The `e2e` job in `.github/workflows/db-migrations.yml` runs this on push to
`master` after the `apply` job has pushed migrations to staging. It needs the
`SUPABASE_SERVICE_KEY` repo secret (the DB URL is built from the existing
`SUPABASE_DB_PASSWORD` secret).
