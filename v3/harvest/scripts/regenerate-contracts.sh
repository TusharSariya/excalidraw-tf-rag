#!/usr/bin/env bash
# Regenerate the pydantic model(s) from the canonical JSON Schema contract(s).
#
# The .json under src/harvest/contracts/ is the single source of truth; the
# generated _*.py model is derived and committed. The generator version is
# pinned so regeneration is reproducible — a CI drift guard can run this and
# `git diff --exit-code` to prove the committed model matches the schema.
#
# Run from anywhere:  v3/harvest/scripts/regenerate-contracts.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GEN='datamodel-code-generator==0.68.1'

uvx --from "$GEN" datamodel-codegen \
  --input src/harvest/contracts/seed_message.v1.json \
  --input-file-type jsonschema \
  --output src/harvest/contracts/_seed_message_v1.py \
  --output-model-type pydantic_v2.BaseModel \
  --class-name SeedMessage \
  --disable-timestamp \
  --use-annotated \
  --field-constraints \
  --use-schema-description

# Conform the generated file to the repo's ruff style (the generator emits
# black/isort formatting; the repo lints + formats with ruff).
uvx ruff check --fix src/harvest/contracts/_seed_message_v1.py >/dev/null
uvx ruff format src/harvest/contracts/_seed_message_v1.py >/dev/null

echo "regenerated src/harvest/contracts/_seed_message_v1.py"
