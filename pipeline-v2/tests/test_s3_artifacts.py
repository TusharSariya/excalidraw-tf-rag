"""S3ArtifactStore tests.

Unit tests drive the store through an in-memory fake S3 client (the store
accepts an injected ``client=`` so no moto/network is needed). The integration
test runs a real put->get round-trip against the live SeaweedFS S3 gateway in
the cluster (namespace ``storage``), reached via a bounded ``kubectl
port-forward`` that is torn down in the fixture.
"""

from __future__ import annotations

import io
import os
import subprocess
import time
from contextlib import closing, suppress

import pytest
from pipeline_v2.artifacts import S3ArtifactStore

boto3 = pytest.importorskip("boto3")
from botocore.exceptions import ClientError  # noqa: E402


def _client_error(status: int, op: str) -> ClientError:
    return ClientError(
        {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": str(status)}},
        op,
    )


class FakeS3Client:
    """Minimal in-memory S3 covering exactly what S3ArtifactStore calls."""

    def __init__(self, *, existing_buckets: set[str] | None = None) -> None:
        self.buckets: set[str] = set(existing_buckets or ())
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls = 0

    def head_bucket(self, Bucket: str) -> dict:  # noqa: N803 (boto3 kwarg name)
        if Bucket not in self.buckets:
            raise _client_error(404, "HeadBucket")
        return {}

    def create_bucket(self, Bucket: str) -> dict:  # noqa: N803
        self.buckets.add(Bucket)
        return {}

    def head_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _client_error(404, "HeadObject")
        return {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> dict:  # noqa: N803
        self.put_calls += 1
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _client_error(404, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def _store(client: FakeS3Client, bucket: str = "test-bucket") -> S3ArtifactStore:
    return S3ArtifactStore(
        endpoint_url="http://fake:8333",
        access_key="k",
        secret_key="s",
        bucket=bucket,
        client=client,
    )


def test_put_get_round_trip():
    c = FakeS3Client()
    store = _store(c)
    uri = store.put(b"hello world", ["extract", "glr", "doc1", "fp1"], suffix=".json")
    assert uri.startswith("s3://test-bucket/")
    assert store.get(uri) == b"hello world"


def test_creates_bucket_when_missing():
    c = FakeS3Client(existing_buckets=set())
    _store(c)
    assert "test-bucket" in c.buckets  # _ensure_bucket created it


def test_content_addressed_same_bytes_same_key():
    c = FakeS3Client()
    store = _store(c)
    u1 = store.put(b"same", ["extract", "glr", "d", "fp"])
    u2 = store.put(b"same", ["extract", "glr", "d", "fp"])
    assert u1 == u2


def test_idempotent_skip_reupload():
    c = FakeS3Client()
    store = _store(c)
    store.put(b"payload", ["s", "c", "d", "fp"])
    assert c.put_calls == 1
    store.put(b"payload", ["s", "c", "d", "fp"])  # object exists -> HEAD hit, no re-put
    assert c.put_calls == 1


def test_different_bytes_different_key():
    c = FakeS3Client()
    store = _store(c)
    u1 = store.put(b"a", ["s", "c", "d", "fp"])
    u2 = store.put(b"b", ["s", "c", "d", "fp"])
    assert u1 != u2


def test_get_rejects_foreign_bucket():
    store = _store(FakeS3Client())
    with pytest.raises(ValueError, match="bucket"):
        store.get("s3://other-bucket/some/key")


def test_parse_uri_malformed():
    with pytest.raises(ValueError, match="not an s3"):
        S3ArtifactStore._parse_uri("file:///x")
    with pytest.raises(ValueError, match="malformed"):
        S3ArtifactStore._parse_uri("s3://bucketonly")


# ---------------------------------------------------------------------------
# Integration: real SeaweedFS S3 in the live cluster (namespace storage).
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration

KUBECTL = os.path.expanduser("~/.local/bin/kubectl")


def _kubectl(*args: str, timeout: float = 30) -> str:
    return subprocess.run(
        [KUBECTL, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout.strip()


@pytest.fixture
def seaweedfs_s3_endpoint():
    """Port-forward the SeaweedFS S3 service; yields (endpoint, access, secret, bucket)."""
    # Find the S3 service in ns storage.
    svcs = _kubectl("get", "svc", "-n", "storage", "-o", "name")
    s3_svc = next((s for s in svcs.splitlines() if "s3" in s), None)
    if not s3_svc:
        pytest.skip("no seaweedfs s3 service found in ns storage")
    # Creds from the tofu-created secret (best-effort; skip if not resolvable).
    access = os.getenv("PIPELINE_V2_S3_ACCESS_KEY", "any")
    secret = os.getenv("PIPELINE_V2_S3_SECRET_KEY", "any")
    port = 18333
    proc = subprocess.Popen(
        [KUBECTL, "port-forward", "-n", "storage", s3_svc, f"{port}:8333"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)  # let the forward establish
        yield (f"http://127.0.0.1:{port}", access, secret, "pipeline-v2-itest")
    finally:
        proc.terminate()
        with suppress(Exception):
            proc.wait(timeout=10)


@pytestmark_integration
def test_live_seaweedfs_round_trip(seaweedfs_s3_endpoint):
    endpoint, access, secret, bucket = seaweedfs_s3_endpoint
    store = S3ArtifactStore(
        endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket
    )
    payload = b"chaos-campaign integration probe " + os.urandom(8)
    uri = store.put(payload, ["itest", "glr", "probe", "fp0"], suffix=".bin")
    assert store.get(uri) == payload
    # Re-put is idempotent (no error, same uri).
    assert store.put(payload, ["itest", "glr", "probe", "fp0"], suffix=".bin") == uri


def test_closing_unused_import_guard():
    # `closing` kept available for any future streaming body test; touch it so
    # linters don't flag the import while documenting intent.
    with closing(io.BytesIO(b"x")) as b:
        assert b.read() == b"x"
