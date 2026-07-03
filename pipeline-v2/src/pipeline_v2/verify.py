"""``pipeline-v2 verify`` — the steady-state invariant oracle every chaos
experiment trusts (rearchitecture campaign T0d).

Three independent check layers, run by :func:`run_verify`:

  1. **DATA** (the core) — cross-checks ``work_state`` against the actual v2
     index (LanceDB + tantivy) and the app-level DLQ, for one corpus:

       (a) every ``work_state`` row with ``state='ok'`` at the indexer stage
           has a matching row in the v2 LanceDB table (tantivy too, best
           effort);
       (b) NO row with ``state='degraded'``/``'failed'`` (any stage) is
           present in the index — this is the exact shape of the v1 bug
           that destroyed 2,266 docs: a non-'ok' outcome recorded in one
           place, read back as "done" somewhere else;
       (c) DLQ topic depth (Kafka) reconciles with the count of
           ``state='failed'`` rows per stage;
       (d) no duplicate index rows for the same chunk id within a corpus.

  2. **RUNTIME** — shells out to kubectl for pod restart counts/phases
     across the campaign's namespaces (inference/kafka/storage/pipeline),
     plus an optional ``--since`` restart->Ready recovery-window report.
     Informational: feeds chaos-experiment hypotheses (E4/E5 recovery
     windows), does not by itself gate PASS/FAIL except for pods stuck in a
     genuinely bad phase.

  3. **V1-INTEGRITY** — sha256 + ``ls -la`` snapshot of the v1 tools'
     ingest state (``graph-layout-rag`` / ``rag-literature-rag``
     ``data/indexes/``). ``verify --v1-snapshot-init`` writes the baseline;
     every other invocation diffs against it. ANY drift is the campaign
     ABORT tripwire — v1 data must never change during the v2 chaos
     campaign.

Every check function returns :class:`CheckOutcome` (PASS/FAIL/SKIP) rather
than raising for "infrastructure not reachable" conditions (no Kafka, no
kubectl, no baseline snapshot yet) — those are SKIP, not silently-passing
and not fatal, so the report is always honest about what it did and didn't
verify. The CLI (``pipeline_v2.cli``) is the only place that turns a report
into a process exit code.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import socket
import subprocess
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pipeline_v2.contracts import STAGES, deadletter_topic
from pipeline_v2.handlers.indexer import CHUNKS_TABLE, V2IndexPaths, bm25_doc_ids
from pipeline_v2.workstate import PostgresWorkStateRepository, WorkStateRepository

logger = logging.getLogger(__name__)

Status = Literal["PASS", "FAIL", "SKIP"]

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_KUBECTL = str(Path.home() / ".local" / "bin" / "kubectl")
DEFAULT_NAMESPACES: tuple[str, ...] = ("inference", "kafka", "storage", "pipeline")

_SNAPSHOT_PATH_ENV = "PIPELINE_V2_V1_SNAPSHOT_PATH"
_V1_INGEST_STATE_RELPATHS: tuple[str, str] = (
    "graph-layout-rag/data/indexes/cuda-qwen0.6b-1024/ingest_state.json",
    "rag-literature-rag/data/indexes/cuda-qwen0.6b-1024/ingest_state.json",
)
_V1_INDEX_DIR_RELPATHS: tuple[str, str] = (
    "graph-layout-rag/data/indexes/",
    "rag-literature-rag/data/indexes/",
)


# --------------------------------------------------------------------------
# Report plumbing
# --------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    layer: str
    check: str
    status: Status
    detail: str = ""


@dataclass
class VerifyReport:
    outcomes: list[CheckOutcome] = field(default_factory=list)

    def add(self, outcome: CheckOutcome) -> None:
        self.outcomes.append(outcome)

    def extend(self, outcomes: Sequence[CheckOutcome]) -> None:
        self.outcomes.extend(outcomes)

    @property
    def failed(self) -> list[CheckOutcome]:
        return [o for o in self.outcomes if o.status == "FAIL"]

    @property
    def ok(self) -> bool:
        """True iff nothing FAILed. An all-SKIP report is still "ok" — it is
        the CLI/caller's job to notice a suspiciously empty report; this
        property answers exactly one question: did any check catch a
        violation?"""
        return not self.failed

    def render(self) -> str:
        lines: list[str] = []
        by_layer: dict[str, list[CheckOutcome]] = {}
        for outcome in self.outcomes:
            by_layer.setdefault(outcome.layer, []).append(outcome)
        for layer, outcomes in by_layer.items():
            lines.append(f"== {layer} ==")
            for outcome in outcomes:
                lines.append(f"  [{outcome.status}] {outcome.check}: {outcome.detail}")
        n_pass = sum(1 for o in self.outcomes if o.status == "PASS")
        n_fail = len(self.failed)
        n_skip = sum(1 for o in self.outcomes if o.status == "SKIP")
        lines.append("")
        lines.append(f"TOTAL: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
        lines.append("VERDICT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Layer 1: DATA invariants
# --------------------------------------------------------------------------


def _lance_rows(lance_dir: Path, corpus: str | None = None) -> list[dict]:
    """Raw ``{id, doc_id, corpus}`` rows from the v2 Lance ``chunks`` table.

    Returns ``[]`` if the table doesn't exist yet (nothing indexed for this
    corpus so far is a legitimate, non-error state — e.g. right after
    ``verify`` is run before any doc has cleared the pipeline).
    """
    if not lance_dir.exists():
        return []
    import lancedb

    db = lancedb.connect(str(lance_dir))
    lister = getattr(db, "list_tables", None) or db.table_names
    names = list(getattr(lister(), "tables", lister()))
    if CHUNKS_TABLE not in names:
        return []
    table = db.open_table(CHUNKS_TABLE)
    rows = table.to_arrow().to_pylist()
    if corpus is None:
        return rows
    return [r for r in rows if r.get("corpus") == corpus]


def check_indexable_present_in_index(
    repo: WorkStateRepository, corpus: str, lance_dir: Path, bm25_dir: Path
) -> list[CheckOutcome]:
    """(a) every state='ok' indexer-stage row has a matching row in the index."""
    ok_doc_ids = {r.doc_id for r in repo.list_indexable(corpus=corpus) if r.stage == "indexer"}
    lance_doc_ids = {r["doc_id"] for r in _lance_rows(lance_dir, corpus)}
    missing_lance = sorted(ok_doc_ids - lance_doc_ids)

    outcomes = [
        CheckOutcome(
            "data",
            "ok-rows-present-in-lance",
            "FAIL" if missing_lance else "PASS",
            (
                f"{len(missing_lance)} of {len(ok_doc_ids)} state='ok' indexer doc(s) missing "
                f"from LanceDB: {missing_lance[:10]}"
                if missing_lance
                else f"{len(ok_doc_ids)} state='ok' indexer doc(s), all present in LanceDB"
            ),
        )
    ]

    try:
        bm25_ids = bm25_doc_ids(bm25_dir)
    except Exception as exc:  # noqa: BLE001 - tantivy read is best-effort
        outcomes.append(
            CheckOutcome(
                "data", "ok-rows-present-in-tantivy", "SKIP", f"tantivy read failed: {exc}"
            )
        )
    else:
        missing_bm25 = sorted(ok_doc_ids - bm25_ids)
        outcomes.append(
            CheckOutcome(
                "data",
                "ok-rows-present-in-tantivy",
                "FAIL" if missing_bm25 else "PASS",
                (
                    f"{len(missing_bm25)} of {len(ok_doc_ids)} state='ok' indexer doc(s) missing "
                    f"from tantivy: {missing_bm25[:10]}"
                    if missing_bm25
                    else f"{len(ok_doc_ids)} state='ok' indexer doc(s), all present in tantivy"
                ),
            )
        )
    return outcomes


def check_no_dead_letters_indexed(
    repo: WorkStateRepository, corpus: str, lance_dir: Path
) -> CheckOutcome:
    """(b) NO degraded/failed row (any stage) is present in the index.

    This is the 2,266-doc bug class: a non-'ok' outcome recorded in
    ``work_state`` but the doc got indexed anyway because something else
    read "row exists" as "done" instead of checking ``state``.
    """
    dead_doc_ids = {r.doc_id for r in repo.get_dead_letters(corpus=corpus)}
    lance_doc_ids = {r["doc_id"] for r in _lance_rows(lance_dir, corpus)}
    bad = sorted(dead_doc_ids & lance_doc_ids)
    if bad:
        return CheckOutcome(
            "data",
            "no-dead-letters-in-index",
            "FAIL",
            f"{len(bad)} doc(s) with degraded/failed work_state ARE present in the v2 index "
            f"(the 2,266-doc bug class): {bad[:10]}",
        )
    return CheckOutcome(
        "data",
        "no-dead-letters-in-index",
        "PASS",
        f"{len(dead_doc_ids)} degraded/failed doc(s) in work_state, none present in the index",
    )


def check_no_duplicate_index_rows(lance_dir: Path, corpus: str) -> CheckOutcome:
    """(d) no duplicate index rows for the same (doc_id, corpus) chunk."""
    rows = _lance_rows(lance_dir, corpus)
    counts = Counter(r["id"] for r in rows)
    dupes = {chunk_id: n for chunk_id, n in counts.items() if n > 1}
    if dupes:
        return CheckOutcome(
            "data",
            "no-duplicate-index-rows",
            "FAIL",
            f"{len(dupes)} duplicate chunk id(s) in LanceDB for corpus={corpus!r}: "
            f"{list(dupes.items())[:10]}",
        )
    return CheckOutcome(
        "data",
        "no-duplicate-index-rows",
        "PASS",
        f"{len(rows)} chunk row(s) for corpus={corpus!r}, no duplicates",
    )


def _kafka_reachable(bootstrap_servers: str, timeout: float = 3.0) -> bool:
    first = bootstrap_servers.split(",")[0].strip()
    host, _, port_str = first.partition(":")
    try:
        port = int(port_str) if port_str else 9092
    except ValueError:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _topic_depth(bootstrap_servers: str, topic: str, timeout: float) -> int | None:
    """Sum of (high - low) watermark offsets across every partition of ``topic``.

    Returns ``None`` if the topic doesn't exist yet (no failures have ever
    been dead-lettered for this stage — a legitimate state, not a check
    failure).
    """
    from confluent_kafka import Consumer, TopicPartition
    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(topic=topic, timeout=timeout)
    topic_meta = metadata.topics.get(topic)
    if topic_meta is None or topic_meta.error is not None:
        return None

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"pipeline-v2-verify-{uuid.uuid4().hex[:8]}",
            "enable.auto.commit": False,
        }
    )
    try:
        total = 0
        for partition_id in topic_meta.partitions:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition_id), timeout=timeout, cached=False
            )
            total += max(high - low, 0)
        return total
    finally:
        consumer.close()


def check_dlq_reconciliation(
    repo: WorkStateRepository,
    corpus: str,
    bootstrap_servers: str,
    stages: Sequence[str] = STAGES,
    timeout: float = 5.0,
) -> list[CheckOutcome]:
    """(c) DLQ topic depth reconciles with the count of state='failed' rows.

    Only ``state='failed'`` rows correspond to an actual DLQ send
    (:class:`pipeline_v2.deadletter.DeadLetterProducer` only ever upserts
    ``'failed'``; ``'degraded'`` outcomes never go through the DLQ). A
    ``deadletter.<stage>`` topic is stage-scoped, not corpus-scoped, so the
    invariant checked is: the number of distinct failed doc_ids in
    work_state for this corpus/stage can never exceed the current message
    depth of that topic (every failed row implies at least one DLQ send
    happened for that doc_id at some point).
    """
    if not _kafka_reachable(bootstrap_servers, timeout=min(timeout, 3.0)):
        return [
            CheckOutcome(
                "data",
                "dlq-reconciliation",
                "SKIP",
                f"kafka at {bootstrap_servers!r} unreachable; DLQ depth not verified",
            )
        ]

    outcomes: list[CheckOutcome] = []
    for stage in stages:
        topic = deadletter_topic(stage)
        failed_doc_ids = {
            r.doc_id
            for r in repo.get_dead_letters(stage=stage, corpus=corpus)
            if r.state == "failed"
        }
        try:
            depth = _topic_depth(bootstrap_servers, topic, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - treat any kafka client error as SKIP
            outcomes.append(
                CheckOutcome(
                    "data",
                    f"dlq-reconciliation:{stage}",
                    "SKIP",
                    f"could not read {topic!r}: {exc}",
                )
            )
            continue
        if depth is None:
            status = "FAIL" if failed_doc_ids else "PASS"
            outcomes.append(
                CheckOutcome(
                    "data",
                    f"dlq-reconciliation:{stage}",
                    status,
                    f"topic {topic!r} does not exist yet; {len(failed_doc_ids)} failed doc(s) in "
                    f"work_state for corpus={corpus!r}"
                    + (" -- MISSING DLQ EVIDENCE" if failed_doc_ids else ""),
                )
            )
            continue
        if len(failed_doc_ids) > depth:
            outcomes.append(
                CheckOutcome(
                    "data",
                    f"dlq-reconciliation:{stage}",
                    "FAIL",
                    f"{len(failed_doc_ids)} distinct failed doc(s) in work_state but only {depth} "
                    f"message(s) currently in {topic!r}",
                )
            )
        else:
            outcomes.append(
                CheckOutcome(
                    "data",
                    f"dlq-reconciliation:{stage}",
                    "PASS",
                    f"{len(failed_doc_ids)} distinct failed doc(s) <= {depth} message(s) currently "
                    f"in {topic!r}",
                )
            )
    return outcomes


def run_data_checks(
    *,
    repo: WorkStateRepository,
    corpus: str,
    lance_dir: Path,
    bm25_dir: Path,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    check_dlq: bool = True,
) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    outcomes.extend(check_indexable_present_in_index(repo, corpus, lance_dir, bm25_dir))
    outcomes.append(check_no_dead_letters_indexed(repo, corpus, lance_dir))
    outcomes.append(check_no_duplicate_index_rows(lance_dir, corpus))
    if check_dlq:
        outcomes.extend(check_dlq_reconciliation(repo, corpus, bootstrap_servers))
    return outcomes


# --------------------------------------------------------------------------
# Layer 2: RUNTIME (kubectl)
# --------------------------------------------------------------------------


def _kubectl_get_pods_json(kubectl: str, namespace: str, timeout: float) -> dict | None:
    try:
        proc = subprocess.run(
            [kubectl, "get", "pods", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def run_runtime_checks(
    *,
    kubectl: str = DEFAULT_KUBECTL,
    namespaces: Sequence[str] = DEFAULT_NAMESPACES,
    since: str | None = None,
    timeout: float = 30.0,
) -> list[CheckOutcome]:
    """Pod restart counts + phases per namespace, plus an optional recovery window.

    Informational by design (feeds E4/E5 recovery-window hypotheses): a
    non-Running phase or a container stuck in CrashLoopBackOff is a FAIL;
    high restart *counts* alone are reported, not failed on, since chaos
    experiments deliberately kill pods and a healthy Running pod with
    restart history is exactly the recovered state we want to see.
    """
    outcomes: list[CheckOutcome] = []
    for namespace in namespaces:
        data = _kubectl_get_pods_json(kubectl, namespace, timeout)
        if data is None:
            outcomes.append(
                CheckOutcome(
                    "runtime",
                    f"pods:{namespace}",
                    "SKIP",
                    "kubectl unavailable or `get pods` failed (no cluster reachable?)",
                )
            )
            continue
        items = data.get("items", [])
        if not items:
            outcomes.append(
                CheckOutcome("runtime", f"pods:{namespace}", "PASS", "no pods in namespace")
            )
            continue
        bad: list[str] = []
        detail_lines: list[str] = []
        for pod in items:
            name = pod.get("metadata", {}).get("name", "?")
            status = pod.get("status", {})
            phase = status.get("phase", "Unknown")
            container_statuses = status.get("containerStatuses", [])
            restarts = sum(cs.get("restartCount", 0) for cs in container_statuses)
            crash_looping = any(
                cs.get("state", {}).get("waiting", {}).get("reason") == "CrashLoopBackOff"
                for cs in container_statuses
            )
            detail_lines.append(f"{name} phase={phase} restarts={restarts}")
            if phase in ("Failed", "Unknown") or crash_looping:
                bad.append(name)
        status_val: Status = "FAIL" if bad else "PASS"
        detail = "; ".join(detail_lines)
        if bad:
            detail += f" -- unhealthy: {bad}"
        outcomes.append(CheckOutcome("runtime", f"pods:{namespace}", status_val, detail))

    if since is not None:
        outcomes.append(_recovery_window(kubectl, namespaces, since, timeout))
    return outcomes


def _recovery_window(
    kubectl: str, namespaces: Sequence[str], since: str, timeout: float
) -> CheckOutcome:
    """Time from a pod's last restart to its Ready condition, for pods restarted since ``since``."""
    lines: list[str] = []
    for namespace in namespaces:
        data = _kubectl_get_pods_json(kubectl, namespace, timeout)
        if data is None:
            continue
        for pod in data.get("items", []):
            name = pod.get("metadata", {}).get("name", "?")
            status = pod.get("status", {})
            ready_time = next(
                (
                    c.get("lastTransitionTime")
                    for c in status.get("conditions", [])
                    if c.get("type") == "Ready" and c.get("status") == "True"
                ),
                None,
            )
            last_restart = None
            for cs in status.get("containerStatuses", []):
                finished_at = cs.get("lastState", {}).get("terminated", {}).get("finishedAt")
                if finished_at and (last_restart is None or finished_at > last_restart):
                    last_restart = finished_at
            if last_restart is None or last_restart < since:
                continue
            if ready_time is None:
                lines.append(f"{namespace}/{name}: restarted at {last_restart}, not yet Ready")
                continue
            window = _iso_seconds_between(last_restart, ready_time)
            window_str = f"{window:.1f}s" if window is not None else "unparseable timestamps"
            lines.append(
                f"{namespace}/{name}: restart->ready in {window_str} (restart={last_restart}, ready={ready_time})"
            )
    detail = "; ".join(lines) if lines else f"no pod restarts observed since {since}"
    return CheckOutcome("runtime", "recovery-window", "PASS", detail)


def _iso_seconds_between(start: str, end: str) -> float | None:
    try:
        dt_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        dt_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt_end - dt_start).total_seconds()


# --------------------------------------------------------------------------
# Layer 3: V1-INTEGRITY (campaign ABORT tripwire)
# --------------------------------------------------------------------------


def _repo_root() -> Path:
    # verify.py is pipeline-v2/src/pipeline_v2/verify.py -> repo root is 3 up.
    return Path(__file__).resolve().parents[3]


def default_snapshot_path(repo_root: Path | None = None) -> Path:
    raw = os.getenv(_SNAPSHOT_PATH_ENV, "").strip()
    if raw:
        return Path(raw)
    root = repo_root or _repo_root()
    return root / "pipeline-v2" / "data" / "v1-snapshot.txt"


def build_v1_snapshot(repo_root: Path | None = None) -> str:
    """sha256 of both tools' ingest_state.json + ``ls -la`` of both indexes/ dirs.

    Shells out to the real ``sha256sum``-equivalent (hashlib) and ``ls -la``
    so the on-disk format matches exactly what an operator would get running
    those commands by hand — this is a human-diffable snapshot, not a
    private serialization format.
    """
    root = repo_root or _repo_root()
    lines: list[str] = []
    for relpath in _V1_INGEST_STATE_RELPATHS:
        path = root / relpath
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {relpath}")
        else:
            lines.append(f"MISSING  {relpath}")

    ls_proc = subprocess.run(
        ["ls", "-la", *_V1_INDEX_DIR_RELPATHS],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root,
    )
    ls_output = ls_proc.stdout if ls_proc.returncode == 0 else f"ls -la failed: {ls_proc.stderr}"
    lines.append(ls_output.rstrip("\n"))
    return "\n".join(lines) + "\n"


def write_v1_snapshot(path: Path, repo_root: Path | None = None) -> str:
    content = build_v1_snapshot(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return content


def check_v1_integrity(snapshot_path: Path, repo_root: Path | None = None) -> CheckOutcome:
    if not snapshot_path.exists():
        return CheckOutcome(
            "v1-integrity",
            "v1-snapshot-diff",
            "SKIP",
            f"no baseline snapshot at {snapshot_path}; run `verify --v1-snapshot-init` first",
        )
    baseline = snapshot_path.read_text(encoding="utf-8")
    current = build_v1_snapshot(repo_root)
    if baseline == current:
        return CheckOutcome(
            "v1-integrity",
            "v1-snapshot-diff",
            "PASS",
            f"v1 index state matches baseline ({snapshot_path})",
        )
    diff = list(
        difflib.unified_diff(
            baseline.splitlines(), current.splitlines(), "baseline", "current", lineterm=""
        )
    )
    return CheckOutcome(
        "v1-integrity",
        "v1-snapshot-diff",
        "FAIL",
        "V1 DATA DRIFT DETECTED -- CAMPAIGN ABORT TRIPWIRE. v1 data must never change during "
        "the v2 chaos campaign. diff: " + " | ".join(diff[:40]),
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_verify(
    *,
    corpus: str | None,
    database_url: str | None = None,
    indexes_root: Path | None = None,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    check_dlq: bool = True,
    kubectl: str = DEFAULT_KUBECTL,
    namespaces: Sequence[str] = DEFAULT_NAMESPACES,
    since: str | None = None,
    skip_runtime: bool = False,
    snapshot_path: Path | None = None,
    skip_v1: bool = False,
    repo: WorkStateRepository | None = None,
) -> VerifyReport:
    """Run whichever layers are applicable and return the combined report.

    ``corpus is None`` skips the DATA layer entirely (useful for a
    runtime-only or v1-only sanity check); everything else runs
    unconditionally unless explicitly skipped.
    """
    report = VerifyReport()

    if corpus is not None:
        data_repo = repo if repo is not None else PostgresWorkStateRepository(database_url)
        paths = V2IndexPaths.for_corpus(corpus, root=indexes_root)
        report.extend(
            run_data_checks(
                repo=data_repo,
                corpus=corpus,
                lance_dir=paths.lance_dir,
                bm25_dir=paths.bm25_dir,
                bootstrap_servers=bootstrap_servers,
                check_dlq=check_dlq,
            )
        )

    if not skip_runtime:
        report.extend(run_runtime_checks(kubectl=kubectl, namespaces=namespaces, since=since))

    if not skip_v1:
        report.add(check_v1_integrity(snapshot_path or default_snapshot_path()))

    return report
