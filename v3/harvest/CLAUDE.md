# harvest — agent guidance

On push, follow the repo-wide **Versioned packages & changelogs** policy in the
root `CLAUDE.md` — bump `version` (+ `uv lock`), add a `CHANGELOG.md` entry, and
update this `README.md` on architecture/usage changes. Design rationale lives in
`docs/architecture/harvest-*.md` — update those for design changes too.

## Contracts are generated

`src/harvest/contracts/_seed_message_v1.py` is generated from
`seed_message.v1.json` — never hand-edit it. Change the `.json`, then run
`scripts/regenerate-contracts.sh`, and keep the in-DB copy
(`public.seed_message_schema()` in the migrations) byte-identical.
