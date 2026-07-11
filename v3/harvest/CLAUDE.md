# harvest — agent guidance

When you change this package **and push**, keep three things in sync:

1. **Bump `version` in `pyproject.toml`** (then `uv lock`). The `build-push` CI
   guard rejects re-publishing an image tag that already exists in prod ECR, so
   any change that ships in the image needs a new version. Prod publishes on merge
   to `master`, so once a version is merged it is permanently taken — bump for the
   next change.
2. **Add a `CHANGELOG.md` entry** under that version describing what changed
   (Added / Changed / Fixed).
3. **Update `README.md`** whenever the architecture, configuration, or usage
   changes.

The version in `pyproject.toml`, the top `CHANGELOG.md` entry, and the README must
agree. Design rationale lives in `docs/architecture/harvest-*.md` — update those
for design changes, not just this README.

## Contracts are generated

`src/harvest/contracts/_seed_message_v1.py` is generated from
`seed_message.v1.json` — never hand-edit it. Change the `.json`, then run
`scripts/regenerate-contracts.sh`, and keep the in-DB copy
(`public.seed_message_schema()` in the migrations) byte-identical.
