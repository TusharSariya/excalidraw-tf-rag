from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import click

from pipeline_v2 import manifest_migrate, verify
from pipeline_v2.workstate import PostgresWorkStateRepository


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
