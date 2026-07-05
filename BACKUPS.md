# Backups — where everything lives (read this before assuming data is lost)

This desktop (`tushar-ubuntu`) was wiped after 2026-07-05. All irreplaceable
local-only data was backed up to S3 first. **Nothing needs re-harvesting or
re-judging from scratch — restore from S3.**

## S3 location

| | |
|---|---|
| Bucket | `s3://rag-backups-992382747916-us-east-1` |
| Region | `us-east-1` |
| Account | `992382747916` |
| Profile (desktop) | `rag-backup` (IAM `user/service/rag-backup`, scoped to this bucket — cannot ListAllBuckets, that's expected) |
| Profile (Mac) | `admin` |
| Bucket source of truth | `aws-account-infra` → `tofu -chdir=stacks/rag output -raw backup_bucket_name` (or `export RAG_BACKUP_BUCKET=rag-backups-992382747916-us-east-1`) |
| Catalog (dynamo) | `rag-backup-catalog` — in-bucket catalog at `_catalog/runs/<run-id>.json` is the durable index of every run |

## Latest full backup

- **Run-id:** `20260705T171953Z-tushar-ubuntu` (2026-07-05, after the eval-trust rebuild + de-biased baseline + Phase-2 bake-offs).
- Prior full run: `20260630T040046Z-tushar-ubuntu`.
- Runs are **immutable and run-stamped** — a new backup never overwrites an old one; unchanged content is deduped against the prior run via the catalog.

## What is backed up (per tool: `graph-layout-rag`, `rag-literature-rag`)

- `corpus/raw/` — the harvested PDF corpus (GLACIER_IR, `aws s3 sync`)
- `indexes/<embed-profile>/<run-id>/archive.tar.zst` — built LanceDB+BM25+sparse indexes (incl. tonight's `cuda-qwen0.6b-1024`)
- `eval/eval/<run-id>/archive.tar.zst` — **the judged qrels + gold sets** (the DeepSeek-judged complete-pool relevance labels — expensive to regenerate)
- `extracted/<model>/` — docling/raptor extraction caches
- `related/<model>/`, `meta/<run-id>/` — manifest.json, citations.sqlite, harvest DBs, ingest state
- **Excluded (regenerable, NOT backed up):** `embed_cache/`, `logs/`, `contextual_cache.json`, `*.log/.bak` — these rebuild from the extract cache + models.

## Repo + code state

- The `excalidraw-tf-rag` git repo is on GitHub (`TusharSariyaOrg/excalidraw-tf-rag`); branch `research-tool-e0-e1` is pushed (PR #7). All commits on all branches are pushed — nothing unpushed.
- **Safety net** for the 1.9 MB of *uncommitted* working-tree changes (pre-existing `pipeline-v2`/`deploy`/`v3` work + untracked `scripts/validate_skills.py` + `.agents/skills/python-architecture/`) is at:
  `s3://…/repo-state/20260705T171953Z-tushar-ubuntu/` → `excalidraw-tf-rag-allrefs.bundle` (all git refs) + `excalidraw-uncommitted.tar.gz` (the working-tree diff/untracked).

## NOT backed up (regenerable on demand)

- `~/.ollama` (~5.8 GB) and `~/.cache/huggingface` (~19 GB) — model weights, re-downloadable (`ollama pull gemma3n:e2b`, HF auto-download of `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`).
- `graphify-out/graph.json` — committed in the repo.

## How to restore

```bash
cd rag-common/scripts
export RAG_BACKUP_BUCKET=rag-backups-992382747916-us-east-1 AWS_PROFILE=rag-backup
./restore_from_s3.sh --help                 # options
./restore_from_s3.sh                         # restore latest run of everything
# or a specific run / tool / category — see restore_from_s3.sh
```

Restore repo working state:
```bash
aws s3 cp s3://rag-backups-992382747916-us-east-1/repo-state/20260705T171953Z-tushar-ubuntu/excalidraw-uncommitted.tar.gz - | tar xz
```

> Keep a copy of this note in the `aws-account-infra` repo too (that repo owns the bucket/catalog Terraform and was not checked out on the wiped desktop).
