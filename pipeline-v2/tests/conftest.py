"""Gate integration tests behind RUN_INTEGRATION, mirroring rag-common /
graph-layout-rag / rag-literature-rag (the majority sibling convention; see
gpu-gateway's --gateway-url flag for the other convention used in this repo).

    RUN_INTEGRATION=1 uv run pytest     # everything, incl. a live postgres:16

Unlike the siblings' node-id-seeded set (needed there so pre-existing test
files stay untouched), tests in this package mark themselves directly with
``@pytest.mark.integration`` — this is a new package, so there is no
untouched-file constraint to preserve, and a direct decorator is harder to
let drift out of sync with the actual test.

The ``postgres`` fixture spins up a real ``postgres:16`` container via
``docker run`` on a random free port, session-scoped so every integration
test in the run shares one container, and tears it down at session end.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _cpu_only_extract_stage(monkeypatch):
    """pipeline-v2's extract stage must run CPU-only (see handlers/extract.py's guard).

    This box may have a real CUDA GPU (owned by gpu-gateway); force
    ``CUDA_VISIBLE_DEVICES=""`` for the test process so the guard sees the
    same "no GPU visible" world a correctly-launched extract worker would,
    rather than disabling the guard itself.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs docker (a real postgres:16 container); "
        "skipped unless RUN_INTEGRATION=1",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test (set RUN_INTEGRATION=1 to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """Session-scoped live postgres:16 container, started via `docker run`."""
    if not os.environ.get("RUN_INTEGRATION"):
        pytest.skip("integration test (set RUN_INTEGRATION=1 to run)")

    port = _free_port()
    name = f"pipeline-v2-test-pg-{uuid.uuid4().hex[:10]}"
    dsn = f"postgresql://postgres:postgres@127.0.0.1:{port}/postgres"

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=postgres",
            "-p",
            f"{port}:5432",
            "postgres:16",
        ],
        check=True,
        capture_output=True,
    )

    try:
        _wait_for_postgres(name, timeout=60.0, dsn=dsn)
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _wait_for_postgres(container_name: str, timeout: float, dsn: str | None = None) -> None:
    """Wait for the container to accept real SQL connections.

    The official postgres image restarts itself once after first-time
    initdb (init, then a fresh server start to actually accept
    connections) — `pg_isready` can report ready during that brief restart
    window, right before the connection gets reset. Require two
    consecutive successful real ``psycopg`` connections (not just
    ``pg_isready``) to land on the other side of that restart, not the
    tail end of the pre-restart process.
    """
    import psycopg

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    consecutive_ok = 0
    while time.monotonic() < deadline:
        try:
            if dsn is not None:
                with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
                    conn.execute("SELECT 1")
            else:
                result = subprocess.run(
                    ["docker", "exec", container_name, "pg_isready", "-U", "postgres"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode(errors="replace"))
            consecutive_ok += 1
            if consecutive_ok >= 2:
                return
        except Exception as exc:  # noqa: BLE001 - retry until timeout
            last_err = exc
            consecutive_ok = 0
        time.sleep(0.5)
    raise RuntimeError(
        f"postgres container {container_name!r} did not become ready: {last_err}"
    ) from last_err


@pytest.fixture
def clean_postgres_dsn(postgres_dsn: str) -> str:
    """The shared container's DSN, with work_state/manifest_item truncated first.

    Function-scoped so each test starts from an empty ledger even though the
    container itself is shared across the whole session (starting a fresh
    postgres:16 container per test would dominate the integration run's
    wall-clock time).
    """
    import psycopg

    with psycopg.connect(postgres_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS work_state CASCADE")
        cur.execute("DROP VIEW IF EXISTS indexable_work_state CASCADE")
        cur.execute("DROP TABLE IF EXISTS manifest_item CASCADE")
    return postgres_dsn


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
def repo(request):
    """A ready-to-use WorkStateRepository — contract tests run against both backends.

    "memory" always runs offline. "postgres" is marked integration and pulls
    in the live container via clean_postgres_dsn, so it's only instantiated
    (and only then requires docker) when RUN_INTEGRATION=1.
    """
    from pipeline_v2.workstate import InMemoryWorkStateRepository, PostgresWorkStateRepository

    if request.param == "memory":
        repo = InMemoryWorkStateRepository()
    else:
        dsn = request.getfixturevalue("clean_postgres_dsn")
        repo = PostgresWorkStateRepository(dsn)
    repo.init_db()
    return repo


# --- Kafka (KIP-932 share groups) fixtures -------------------------------
#
# Broker env is the spike's proven config (scratchpad/sharegroup-spike/
# REPORT.md Milestone 2) with one change: group.share.max.record.lock.
# duration.ms raised from the spike's 600000 to 900000 (Lane A3's finding —
# the broker DEFAULTS that max to 60000ms and silently clamps a larger
# client request; 900000 gives real headroom above our 600000ms client
# lock request). Session-scoped: every integration test in the run shares
# one broker; per-test isolation comes from unique topic names + unique
# consumer group ids (see the `kafka_topics` fixture in
# test_messaging_integration.py), not from a fresh container per test —
# starting Kafka once keeps the integration run's wall-clock reasonable.

KAFKA_RECORD_LOCK_MS = 600_000
KAFKA_MAX_RECORD_LOCK_MS = 900_000


@pytest.fixture(scope="session")
def kafka_bootstrap() -> str:
    """Session-scoped live Kafka 4.3.1 (apache/kafka:latest) container via `docker run`.

    IMPORTANT: the broker's PLAINTEXT listener, its advertised listener, and
    the host port mapping must all use the *same* port number. Kafka's
    internal share-coordinator/persister machinery (which initializes each
    share-partition's state — the exact thing `share.auto.offset.reset`
    governs) connects to the broker via its own ADVERTISED_LISTENERS
    address, including from inside the container. If that address names a
    host-mapped port that isn't also the container's internal listener port
    (e.g. internal 9092 + advertised host-port 54321 via `-p 54321:9092`),
    that self-loopback connection fails and every share group's state
    initialization dies with "Exhausted max retries to find coordinator"
    (errorCode 13) — consumers join the group but are never assigned any
    partition, so nothing is ever delivered, with no client-visible error.
    Discovered the hard way debugging this fixture; keep listener/advertised/
    host-port numerically identical if you ever change this.
    """
    if not os.environ.get("RUN_INTEGRATION"):
        pytest.skip("integration test (set RUN_INTEGRATION=1 to run)")

    port = _free_port()
    name = f"pipeline-v2-test-kafka-{uuid.uuid4().hex[:10]}"
    bootstrap = f"localhost:{port}"

    env = {
        "KAFKA_NODE_ID": "1",
        "KAFKA_PROCESS_ROLES": "broker,controller",
        "KAFKA_LISTENERS": f"PLAINTEXT://:{port},CONTROLLER://:9093",
        "KAFKA_ADVERTISED_LISTENERS": f"PLAINTEXT://{bootstrap}",
        "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
        "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
        "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@localhost:9093",
        "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
        "KAFKA_SHARE_COORDINATOR_STATE_TOPIC_REPLICATION_FACTOR": "1",
        "KAFKA_SHARE_COORDINATOR_STATE_TOPIC_MIN_ISR": "1",
        "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
        "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
        "KAFKA_GROUP_COORDINATOR_REBALANCE_PROTOCOLS": "classic,consumer,share",
        "KAFKA_GROUP_SHARE_RECORD_LOCK_DURATION_MS": str(KAFKA_RECORD_LOCK_MS),
        "KAFKA_GROUP_SHARE_MAX_RECORD_LOCK_DURATION_MS": str(KAFKA_MAX_RECORD_LOCK_MS),
        # Fire the DLQ-exhaustion test in a reasonable number of attempts.
        "KAFKA_GROUP_SHARE_DELIVERY_COUNT_LIMIT": "10",
    }
    env_args: list[str] = []
    for key, value in env.items():
        env_args += ["-e", f"{key}={value}"]

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:{port}",
            *env_args,
            "apache/kafka:latest",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    try:
        _wait_for_kafka(name, bootstrap, timeout=90.0)
        yield bootstrap
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _wait_for_kafka(container_name: str, bootstrap: str, timeout: float) -> None:
    from confluent_kafka.admin import AdminClient

    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            admin = AdminClient({"bootstrap.servers": bootstrap})
            metadata = admin.list_topics(timeout=5)
            if metadata.brokers:
                return
        except Exception as exc:  # noqa: BLE001 - retry until timeout
            last_exc = exc
        time.sleep(1.0)
    raise RuntimeError(
        f"kafka container {container_name!r} did not become ready: {last_exc}"
    ) from last_exc
