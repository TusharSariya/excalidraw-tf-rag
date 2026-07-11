# Changelog

Notable changes to the `harvest` package. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions match `pyproject.toml`
(the image tag the `build-push` CI guard checks against prod ECR).

## [0.5.0]

### Added
- `harvest-work` worker — a single-threaded, drain-then-exit pull loop over the
  Supabase `pgmq_public` queue wrappers (`read` → `contracts.parse` → `archive`,
  exit when the queue empties). Proven end-to-end against staging.
- Per-entrypoint settings: a shared base (`SUPABASE_URL` / `SUPABASE_SERVICE_KEY`
  + source API keys, with a fail-fast `client()`) plus `WorkerSettings`
  (`--source`) and `SeederSettings` (`--corpus`/`--arxiv_id`/`--doi`).

### Changed
- README rewritten for the seed → queue → worker architecture.
- Settings drop the legacy REST url/key and the direct-pg DSN in favour of
  `supabase_url` / `supabase_service_key`.

## [0.4.0]

### Added
- Seed → message contract in `contracts/`: a versioned JSON Schema
  (`seed_message.v1.json`), a **generated** `SeedMessage` pydantic model, and a
  tolerant-reader `parse()` — additive fields warn + are ignored;
  subtractive/mistyped/unrecognized-version messages raise.
- DB enforcement (migration `20260711040000`): `publish_seed` stamps
  `{"v":1,…}` and `pg_jsonschema`-validates every message before enqueuing.

## [0.3.0]

- Baseline: legacy one-shot harvester (arXiv + Semantic Scholar fetch over the
  Supabase REST client). Superseded by the decoupled seed / queue / worker design.
