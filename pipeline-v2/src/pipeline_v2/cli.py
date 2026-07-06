from __future__ import annotations

import hashlib
import os
import sys
import uuid
from pathlib import Path

import click

from pipeline_v2 import manifest_migrate, verify
from pipeline_v2.workstate import PostgresWorkStateRepository

DEFAULT_ENQUEUE_STAGE_FINGERPRINT = "enqueue-v1"


@click.group()
def main() -> None:
    """pipeline-v2 — work-state ledger + manifest migration (rearchitecture P1)."""


@main.group("workstate")
def workstate_group() -> None:
    """Work-state ledger commands."""


@workstate_group.command("init-db")
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
def init_db_cmd(database_url: str | None) -> None:
    """Create/upgrade the work_state + manifest_item schema."""
    repo = PostgresWorkStateRepository(database_url)
    repo.init_db()
    click.echo("schema applied")


@workstate_group.command("retry-dead-letter")
@click.option("--stage", required=True, help="Stage to retry (e.g. embed, extract).")
@click.option("--corpus", default=None, help="Restrict to one corpus (default: all).")
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
def retry_dead_letter_cmd(stage: str, corpus: str | None, database_url: str | None) -> None:
    """Reset dead-letter rows (state degraded/failed) for a stage to pending-retry.

    Bumps ``attempt`` and clears ``reason``/``artifact_uri``/``run_id`` on
    matching rows. State itself is left as degraded/failed — these rows were
    never indexable and remain not-indexable until a fresh attempt records a
    real 'ok' outcome via ``upsert_outcome``. Rows already in state='ok' are
    never touched.
    """
    repo = PostgresWorkStateRepository(database_url)
    touched = repo.retry_dead_letter(stage, corpus=corpus)
    click.echo(f"reset {touched} dead-letter row(s) for stage={stage!r} corpus={corpus!r}")


@main.group("manifest")
def manifest_group() -> None:
    """v1 manifest.json migration commands."""


@manifest_group.command("migrate")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--corpus", required=True, help="Corpus namespace for the imported items.")
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="After migrating, export back out and byte-compare against the input.",
)
def manifest_migrate_cmd(
    json_path: Path, corpus: str, database_url: str | None, verify: bool
) -> None:
    """Import a v1 manifest.json into the manifest_item table."""
    manifest = manifest_migrate.load_v1_manifest_json(json_path)
    count = manifest_migrate.migrate_to_db(manifest, corpus=corpus, dsn=database_url)
    click.echo(f"migrated {count} item(s) into corpus={corpus!r}")

    if verify:
        exported = manifest_migrate.export_from_db(corpus=corpus, dsn=database_url)
        report = manifest_migrate.diff_manifests(manifest, exported)
        click.echo(report.summary())
        if report.drift:
            sys.exit(1)


@manifest_group.command("export")
@click.option("--corpus", required=True, help="Corpus namespace to export.")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
def manifest_export_cmd(corpus: str, out_path: Path, database_url: str | None) -> None:
    """Export the manifest_item table for one corpus back to a manifest.json."""
    manifest = manifest_migrate.export_from_db(corpus=corpus, dsn=database_url)
    manifest_migrate.export_to_json_file(manifest, out_path)
    click.echo(f"exported {len(manifest['items'])} item(s) to {out_path}")


@main.command("consume")
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["extract", "embed", "indexer"]),
    help="Pipeline stage to run.",
)
@click.option("--corpus", required=True, help="Corpus namespace this consumer processes.")
@click.option(
    "--queue",
    "queue_backend",
    default="share",
    type=click.Choice(["share", "classic"]),
    help="Queue backend: KIP-932 share group (default) or classic consumer group fallback.",
)
@click.option("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers.")
@click.option("--group-id", default=None, help="Consumer group id (default: pipeline-v2.<stage>).")
@click.option(
    "--max-iterations",
    default=None,
    type=int,
    help="Stop after N poll iterations (omit to run forever; used by tests/bounded runs).",
)
@click.option("--poll-timeout", default=1.0, type=float, help="Seconds per poll() call.")
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
def consume_cmd(
    stage: str,
    corpus: str,
    queue_backend: str,
    bootstrap_servers: str,
    group_id: str | None,
    max_iterations: int | None,
    poll_timeout: float,
    database_url: str | None,
) -> None:
    """Run a StageConsumer loop for one stage/corpus against a real Kafka queue."""
    from pipeline_v2.consumers import STAGE_CONFIG_BUILDERS, StageConsumer
    from pipeline_v2.deadletter import DeadLetterProducer
    from pipeline_v2.messaging import build_queue

    metrics_port = os.environ.get("PIPELINE_V2_METRICS_PORT")
    if metrics_port:
        from pipeline_v2.metrics import start_metrics_server

        start_metrics_server(int(metrics_port))

    repo = PostgresWorkStateRepository(database_url)
    repo.init_db()

    config = STAGE_CONFIG_BUILDERS[stage](repo)
    resolved_group_id = group_id or f"pipeline-v2.{stage}.{corpus}"

    if queue_backend == "share":
        # KIP-932 share groups capture `share.auto.offset.reset` at the group's
        # FIRST attach; the default is `latest`, so without this a fresh
        # consumer silently skips every already-enqueued record (the spike
        # gotcha, docs/rearchitecture-v2-decision-log.md A4). Set earliest
        # BEFORE build_queue creates the first consumer.
        from pipeline_v2.messaging import ensure_share_group_earliest_reset

        ensure_share_group_earliest_reset(bootstrap_servers, resolved_group_id)

    queue = build_queue(
        queue_backend,
        bootstrap_servers=bootstrap_servers,
        group_id=resolved_group_id,
        topics=[config.input_topic],
    )
    dlq = DeadLetterProducer(queue=queue, repo=repo)
    consumer = StageConsumer(
        queue=queue, repo=repo, config=config, dlq=dlq, run_id=str(uuid.uuid4())
    )
    try:
        processed = consumer.run_forever(max_iterations=max_iterations, poll_timeout=poll_timeout)
        click.echo(f"processed {processed} record(s) for stage={stage!r} corpus={corpus!r}")
    finally:
        queue.close()


def _iter_manifest_enqueue_items(manifest: dict) -> list[dict]:
    """Extract (doc_id, pdf_path, sha256, title, source_url) tuples from a v1 manifest dict.

    Only items that carry a ``localPath`` (the on-disk PDF, per the v1
    manifest shape documented in :mod:`pipeline_v2.manifest_migrate`) are
    enqueueable -- metadata-only/failed items have nothing for the extract
    stage to read. ``sha256`` prefers the v1-computed content hash already
    stored on the item (identical semantics to what v1 used for the same
    file) and falls back to hashing the PDF on disk if that field is absent.
    """
    out: list[dict] = []
    for item in manifest.get("items", []):
        local_path = item.get("localPath")
        if not local_path:
            continue
        pdf_path = Path(local_path)
        sha256 = item.get("sha256")
        if not sha256 and pdf_path.is_file():
            sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        out.append(
            {
                "doc_id": item["id"],
                "pdf_path": local_path,
                "sha256": sha256 or "",
                "title": item.get("title") or item["id"],
                "source_url": item.get("url") or "",
            }
        )
    return out


def _sha256_file(path: Path, *, block_size: int = 1 << 20) -> str:
    """Stream-hash a file in ``block_size`` (default 1 MiB) chunks.

    ``Path.read_bytes()`` loads the whole file into memory; the GLR corpus holds
    a 1.1 GB PDF (plus several >50 MB), which OOMs the 1 Gi enqueue pod. Streaming
    keeps peak memory at one block regardless of file size, while producing the
    identical digest to a whole-file hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_glob_enqueue_items(
    pattern: str,
    limit: int | None = None,
    start_offset: int = 0,
    max_file_mb: float | None = None,
) -> list[dict]:
    """Build enqueue items directly from PDFs on disk, doc_id = filename stem.

    ``limit`` short-circuits BEFORE hashing: a glob over a full corpus can match
    thousands of PDFs, and sha256-ing every one only to truncate afterwards
    reads (and can OOM on) the entire corpus for an ``--limit 5`` request.

    ``start_offset`` skips the first N globbed (sorted) files WITHOUT hashing
    them, so a large corpus can be enqueued in bounded windows
    (offset 0/limit 500, offset 500/limit 500, ...) that each stay within the
    enqueue pod's memory + the producer flush window — the single full-corpus
    enqueue OOMs the pod hashing thousands of PDFs at once.

    ``max_file_mb`` skips (and logs) any PDF larger than the cap by ``stat`` size
    BEFORE opening it — a handful of pathological multi-hundred-MB / 1.1 GB PDFs
    would otherwise OOM both this hash and downstream docling extraction. A
    skipped file does NOT consume the ``limit`` budget.
    """
    import glob as glob_module

    cap_bytes = int(max_file_mb * 1024 * 1024) if max_file_mb else None
    out: list[dict] = []
    for idx, raw_path in enumerate(sorted(glob_module.glob(pattern))):
        if idx < start_offset:
            continue
        if limit is not None and len(out) >= limit:
            break
        pdf_path = Path(raw_path)
        if not pdf_path.is_file():
            continue
        if cap_bytes is not None and pdf_path.stat().st_size > cap_bytes:
            click.echo(
                f"enqueue: skipping oversized PDF {pdf_path.name} "
                f"({pdf_path.stat().st_size / 1024 / 1024:.0f} MB > cap {max_file_mb:.0f} MB)",
                err=True,
            )
            continue
        out.append(
            {
                "doc_id": pdf_path.stem,
                "pdf_path": str(pdf_path),
                "sha256": _sha256_file(pdf_path),
                "title": pdf_path.stem,
                "source_url": "",
            }
        )
    return out


@main.command("enqueue")
@click.option("--corpus", required=True, help="Corpus namespace for the enqueued docs.")
@click.option(
    "--limit", default=None, type=int, help="Max number of docs to enqueue (omit for all)."
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="v1 manifest.json to read PDF paths from (mutually exclusive with --pdf-glob).",
)
@click.option(
    "--pdf-glob",
    default=None,
    help="Glob pattern of PDFs to enqueue directly, doc_id = filename stem "
    "(mutually exclusive with --manifest).",
)
@click.option(
    "--stage-fingerprint",
    default=DEFAULT_ENQUEUE_STAGE_FINGERPRINT,
    help="stage_fingerprint stamped on every produced message (same value across a run "
    "makes re-enqueues idempotent no-ops at the extract stage).",
)
@click.option("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers.")
@click.option("--topic", default="docs.enriched", help="Topic to produce to.")
@click.option(
    "--queue",
    "queue_backend",
    default="classic",
    type=click.Choice(["share", "classic"]),
    help="Queue backend used to produce (default: classic -- no share-group side effects "
    "for a produce-only run).",
)
@click.option("--group-id", default=None, help="Queue group id (default: pipeline-v2.enqueue).")
@click.option(
    "--start-offset",
    default=0,
    type=int,
    help="Skip the first N globbed (sorted) PDFs without hashing them, for enqueuing a "
    "large corpus in bounded windows (e.g. --start-offset 500 --limit 500). The single "
    "full-corpus enqueue OOMs the enqueue pod hashing thousands of PDFs at once.",
)
@click.option(
    "--max-file-mb",
    default=None,
    type=float,
    help="Skip (and log to stderr) any PDF larger than this many MB before hashing it. "
    "A handful of pathological multi-hundred-MB / 1.1 GB PDFs otherwise OOM the enqueue "
    "pod's hash and downstream docling extraction. Only applies to --pdf-glob.",
)
def enqueue_cmd(
    corpus: str,
    limit: int | None,
    manifest_path: Path | None,
    pdf_glob: str | None,
    stage_fingerprint: str,
    bootstrap_servers: str,
    topic: str,
    queue_backend: str,
    group_id: str | None,
    start_offset: int,
    max_file_mb: float | None,
) -> None:
    """Produce docs.enriched (or another topic's) messages for a slice of a v1 manifest or PDFs.

    Exactly one of --manifest / --pdf-glob selects the source. Each item
    becomes one StageMessage with payload {"pdf_path", "title",
    "source_url"} -- the shape the extract stage handler
    (:mod:`pipeline_v2.handlers.extract`) expects.
    """
    from pipeline_v2.contracts import StageMessage
    from pipeline_v2.messaging import build_queue

    if bool(manifest_path) == bool(pdf_glob):
        raise click.UsageError("enqueue needs exactly one of --manifest or --pdf-glob")

    if manifest_path is not None:
        manifest = manifest_migrate.load_v1_manifest_json(manifest_path)
        items = _iter_manifest_enqueue_items(manifest)
        items = items[start_offset : (start_offset + limit) if limit is not None else None]
    else:
        # Pass limit + offset INTO the glob iterator so it only hashes the
        # bounded window [start_offset, start_offset+limit) instead of reading
        # the whole corpus then truncating.
        items = _iter_glob_enqueue_items(  # type: ignore[arg-type]
            pdf_glob, limit=limit, start_offset=start_offset, max_file_mb=max_file_mb
        )

    if not items:
        click.echo("no enqueueable items found (nothing produced)")
        return

    resolved_group_id = group_id or "pipeline-v2.enqueue"
    queue = build_queue(
        queue_backend,
        bootstrap_servers=bootstrap_servers,
        group_id=resolved_group_id,
        topics=[topic],
    )
    try:
        for item in items:
            message = StageMessage(
                doc_id=item["doc_id"],
                corpus=corpus,
                content_sha256=item["sha256"],
                stage_fingerprint=stage_fingerprint,
                attempt=1,
                payload={
                    "pdf_path": item["pdf_path"],
                    "title": item["title"],
                    "source_url": item["source_url"],
                },
            )
            queue.produce(topic, message.model_dump_json().encode("utf-8"))
        queue.flush(timeout=30.0)
    finally:
        queue.close()

    click.echo(f"enqueued {len(items)} message(s) to topic={topic!r} corpus={corpus!r}")


@main.command("verify")
@click.option(
    "--corpus", default=None, help="Corpus namespace to check (DATA layer). Omit to skip it."
)
@click.option(
    "--database-url", default=None, help="Postgres DSN (default: $PIPELINE_V2_DATABASE_URL)."
)
@click.option(
    "--indexes-root",
    default=None,
    type=click.Path(path_type=Path),
    help="v2 indexes root (default: $PIPELINE_V2_INDEXES_DIR).",
)
@click.option(
    "--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers (DLQ check)."
)
@click.option("--skip-dlq", is_flag=True, default=False, help="Skip the DLQ topic-depth check.")
@click.option("--kubectl", default=verify.DEFAULT_KUBECTL, help="Path to kubectl binary.")
@click.option(
    "--namespace",
    "namespaces",
    multiple=True,
    default=verify.DEFAULT_NAMESPACES,
    help="Namespace(s) for the RUNTIME layer (repeatable).",
)
@click.option(
    "--since", default=None, help="ISO timestamp; report restart->Ready recovery windows since it."
)
@click.option(
    "--skip-runtime", is_flag=True, default=False, help="Skip the RUNTIME (kubectl) layer."
)
@click.option(
    "--snapshot-path",
    default=None,
    type=click.Path(path_type=Path),
    help="v1-integrity snapshot file (default: pipeline-v2/data/v1-snapshot.txt).",
)
@click.option(
    "--v1-snapshot-init",
    is_flag=True,
    default=False,
    help="Write a new v1-integrity baseline and exit.",
)
@click.option(
    "--v1-check",
    is_flag=True,
    default=False,
    help="Run ONLY the v1-integrity diff against the baseline.",
)
def verify_cmd(
    corpus: str | None,
    database_url: str | None,
    indexes_root: Path | None,
    bootstrap_servers: str,
    skip_dlq: bool,
    kubectl: str,
    namespaces: tuple[str, ...],
    since: str | None,
    skip_runtime: bool,
    snapshot_path: Path | None,
    v1_snapshot_init: bool,
    v1_check: bool,
) -> None:
    """Steady-state invariant oracle: DATA + RUNTIME + V1-INTEGRITY checks."""
    resolved_snapshot_path = snapshot_path or verify.default_snapshot_path()

    if v1_snapshot_init:
        verify.write_v1_snapshot(resolved_snapshot_path)
        click.echo(f"wrote v1-integrity baseline snapshot to {resolved_snapshot_path}")
        return

    if v1_check:
        outcome = verify.check_v1_integrity(resolved_snapshot_path)
        click.echo(f"[{outcome.status}] {outcome.check}: {outcome.detail}")
        if outcome.status == "FAIL":
            sys.exit(1)
        return

    if corpus is None and skip_runtime:
        raise click.UsageError(
            "verify needs --corpus and/or the RUNTIME/V1 layers; nothing to check."
        )

    report = verify.run_verify(
        corpus=corpus,
        database_url=database_url,
        indexes_root=indexes_root,
        bootstrap_servers=bootstrap_servers,
        check_dlq=not skip_dlq,
        kubectl=kubectl,
        namespaces=list(namespaces),
        since=since,
        skip_runtime=skip_runtime,
        snapshot_path=resolved_snapshot_path,
    )
    click.echo(report.render())
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
