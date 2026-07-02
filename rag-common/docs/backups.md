# RAG → S3 backups

Off-machine backup/restore for the RAG tools' local-only `data/` (indexes, corpus,
docling extract, eval history, metadata) so the local copy can be deleted and rebuilt
with confidence. The destination bucket is provisioned by the `aws-account-infra` repo
(`stacks/rag` → `rag-backups-992382747916-us-east-1`): versioned, SSE-KMS,
lifecycle-tiered. The same stack provisions the **`rag-backup-catalog`** DynamoDB table.

Derived artifacts (indexes, retrieval-indexes, related, extracted, eval, meta) are
**immutable and run-stamped**: each backup writes under a per-run id, nothing is
overwritten, and unchanged artifacts are **deduped** (a new catalog row points at the
prior object instead of re-uploading). Corpus is content-addressed incremental `s3 sync`.

## TL;DR

```bash
# Desktop (authoritative: corpus + cuda/gemini/jina indexes) — run on the box:
AWS_PROFILE=rag-backup RAG_BACKUP_BUCKET=rag-backups-992382747916-us-east-1 \
  bash ~/excalidraw-tf/tools/rag-common/scripts/backup_to_s3.sh

# Mac (owns only the mlx-* builds the GPU box can't make), run LAST:
yarn rag:backup --only mlx

# Verify the backup is complete + current (a pre-nuke gate; exits non-zero on any gap):
yarn rag:verify
yarn rag:verify --drill rag-literature-rag/indexes/mlx-qwen4b   # prove one artifact round-trips

# Restore latest run of an entire tool onto an empty machine:
yarn rag:restore rag-literature-rag
yarn rag:restore graph-layout-rag cuda-qwen0.6b-1024  # one artifact
yarn rag:restore rag-literature-rag --as-of 20260630T0400Z      # point-in-time
```

`--dry-run` prints the plan and touches nothing. `yarn rag:{backup,restore,verify}` are
thin wrappers over `tools/rag-common/scripts/*`.

## Where things live & who runs what

Neither machine is a superset, so backups run on **both** and compose into one bucket
(per-profile S3 keys make collisions impossible):

| Data | Authoritative on | Backed up from |
|---|---|---|
| corpus (`raw/`), `cuda-*` / `gemini-*` / `jina-*` indexes, retrieval-indexes, extract, eval, meta | **desktop** (GPU box) | desktop, plain `backup_to_s3.sh` |
| `mlx-*` indexes (Apple-Silicon-only builds) | **Mac** | Mac, `--only mlx` (run last) |

> ⚠️ Run the **Mac** backup with `--only mlx`, and run it after the desktop. A bare run
> on the Mac would push its (possibly stale) `cuda-*` copies; the desktop is the source
> of truth for everything except `mlx-*`.

## S3 key layout (the collision-proof, run-stamped contract)

```
<tool>/corpus/raw/...                                  # aws s3 sync, GLACIER_IR (NOT run-stamped)
<tool>/indexes/<embed-profile>/<run-id>/archive.tar.zst   # tar+zstd, STANDARD_IA → GLACIER@180d
<tool>/retrieval-indexes/<model>/<run-id>/archive.tar.zst
<tool>/related/<model>/<run-id>/archive.tar.zst
<tool>/extracted/<extraction-model>/<run-id>/archive.tar.zst
<tool>/eval/<name>/<run-id>/archive.tar.zst
<tool>/meta/<run-id>/<file>                            # aws s3 cp, STANDARD
_catalog/runs/<run-id>.json                            # the self-contained run manifest
```

`<run-id>` = `<UTC compact>-<host>` (e.g. `20260630T034531Z-tushars-MacBook-Pro`), minted
once per `backup_to_s3.sh` invocation. The **artifact id** (catalog PK) is the run-id-free
key `<tool>/<category>/<name>`. Uniqueness is structural ⇒ no collisions:

- **New embed profile** → new `indexes/<name>/` automatically (manifest globs `indexes/*`).
  `mlx-qwen4b` and `cuda-qwen0.6b-1024` are distinct artifacts — they never overwrite.
- **New extraction model** → add a line under `[tool.<x>.extracted]` in the manifest.
- **New RAG tool** → add a `[tool.<name>]` block. Nothing else changes.

Each tar/meta object carries its `data/`-relative path in S3 object **metadata**
(`localrel`), so restore rebuilds a machine whose `data/` is empty — it reads the bucket,
not the local disk. Run-stamped objects are tagged `lifecycle-class=run-stamped`, which is
what the GLACIER transition rule keys on (corpus and `_catalog/` are untagged and stay put).

## Catalog: "what existed at time T, and with what config"

Two layers, and **restore never depends on DynamoDB**:

- **In-bucket `_catalog/runs/<run-id>.json`** — the source of truth. One self-contained
  object per run listing every artifact with its `s3_key`, `sha256`, `size_bytes`,
  `content_sig`, storage class, and provenance (build config from `ingest_status.json`,
  tool git sha, host, timestamp).
- **`rag-backup-catalog` DynamoDB table** — a queryable index the harness dual-writes
  (PK `artifact`, SK `run_id`; GSI `by-run`). Used for fast "latest / as-of-T per
  artifact" lookups and for the **dedup check** (compare the live `content_sig` to the
  newest row; on a match, skip the upload and just point a new row at the existing object).
  If the table is unreachable, the backup still succeeds (the run manifest is authoritative)
  and `verify`/`restore` fall back to bucket-only logic.

`content_sig` is a fast metadata-only signature (sha256 over sorted relpath+size+mtime),
so "did this artifact change?" never re-reads file contents.

## Verify (the nuke-safety gate)

`yarn rag:verify` checks, for every artifact present locally, that the **latest** backup is
**FRESH** (its `content_sig` matches disk now) and its S3 object still exists. It prints
`PASS` / `STALE` / `MISSING` per artifact and exits non-zero on any `STALE`/`MISSING` — run
it before trusting a nuke. `--drill <artifact>` downloads one artifact's latest tarball to a
scratch dir, unpacks it, and checksum-compares against the live copy to prove the full
S3 → disk round-trip.

## Configuration

- **Bucket:** `$RAG_BACKUP_BUCKET`, else `tofu output -raw backup_bucket_name` from
  `aws-account-infra` (set `$RAG_AWS_INFRA_DIR` if not at `~/Projects/aws-account-infra`).
  The desktop has no tofu state → set `RAG_BACKUP_BUCKET` there.
- **Catalog table:** `$RAG_BACKUP_CATALOG_TABLE`, else `tofu output -raw
  backup_catalog_table_name`, else the default `rag-backup-catalog`. `--no-catalog`
  skips it; `--no-dedup` forces re-upload.
- **Compression:** `$RAG_ZSTD_LEVEL` (default `-19 --long=27`). The vector indexes and PDF
  corpus are near-incompressible, so the slower `--ultra -22` levels mostly burn CPU; set
  `RAG_ZSTD_LEVEL="--ultra -22 --long=27"` only if you want to chase the last few percent.
- **Creds:** `$AWS_PROFILE` — `rag-backup` on the desktop (least-privilege IAM user scoped
  to *only* this bucket + its KMS key + the catalog table), `admin` on the Mac.
- **aws-cli path:** `$RAG_AWS_BIN` (defaults to `aws` on PATH, else `~/.local/bin/aws`).
- **Manifest:** [`backup-manifest.toml`](../backup-manifest.toml) — per-tool sources +
  the `defaults.exclude` list (`embed_cache`, logs, `*.bak`, …).

## Cost

~30 GB stored across all three tools → **a few $/month** (corpus on GLACIER_IR, tarballs on
STANDARD_IA tiering to GLACIER at 180d, meta on STANDARD; plus versioned history). Dedup
keeps re-runs near-free — unchanged profiles add a catalog row, not a copy. Restore egress
is ~$0.09/GB (a full 30 GB restore ≈ $2.70; one profile is pennies). GLACIER-tiered
artifacts (>180d) need a thaw before download — the `rag-backup` user has `s3:RestoreObject`.

## Lifecycle / pruning

Run-stamped objects transition to GLACIER at 180d but are **never auto-expired**: with
content-addressed dedup, the latest catalog pointer for an unchanged profile can reference a
physically older object, so blind age-based expiry would delete a still-current artifact.
Pruning old runs is a future catalog-aware step (delete runs no `latest`/`as-of` pointer
needs), not an S3 lifecycle expiry.

## Scheduling (later)

The scripts are cron/launchd-safe (explicit `aws` path, no interactive assumptions). To
automate the desktop's run, drop a cron entry / systemd timer that calls `backup_to_s3.sh`
with `AWS_PROFILE=rag-backup` and `RAG_BACKUP_BUCKET=…` set, then `verify_backup.sh` as a
post-check.
```
