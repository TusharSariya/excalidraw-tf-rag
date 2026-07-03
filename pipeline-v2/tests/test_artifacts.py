"""Unit + integration coverage for the artifact store backends.

The offline (unit) tests use moto's ``mock_aws`` to fake AWS S3 semantics
via a real ``boto3`` client injected into :class:`S3ArtifactStore` — moto
only intercepts requests against *default* AWS endpoints, so tests here
construct the mocked boto3 client without a custom ``endpoint_url`` and pass
it in via ``S3ArtifactStore(client=...)`` rather than exercising the
``endpoint_url``/path-style-addressing wiring itself (that needs a real or
SeaweedFS-like S3-compatible server, which is what the
``@pytest.mark.integration`` test at the bottom does against the live
in-cluster SeaweedFS S3 gateway).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time

import pytest
from pipeline_v2.artifacts import (
    LocalArtifactStore,
    S3ArtifactStore,
    artifact_store_from_env,
)

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")


@pytest.fixture
def s3_store():
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        store = S3ArtifactStore(
            endpoint_url="http://unused.invalid",
            access_key="unused",
            secret_key="unused",
            bucket="pipeline-v2-test",
            client=client,
        )
        yield store


def test_ensure_bucket_creates_missing_bucket(s3_store):
    # __init__ already ran _ensure_bucket(); head_bucket should now succeed.
    s3_store._client.head_bucket(Bucket=s3_store.bucket)


def test_put_returns_content_addressed_uri(s3_store):
    content = b"hello seaweedfs"
    uri = s3_store.put(content, ["extract", "corpusA", "doc1"], suffix=".json")

    sha256 = hashlib.sha256(content).hexdigest()
    assert uri.startswith("s3://pipeline-v2-test/")
    assert uri.endswith(f"{sha256}.json")
    assert f"extract/corpusA/doc1/{sha256[:2]}/{sha256}.json" in uri


def test_put_then_get_roundtrip(s3_store):
    content = b"round trip bytes \x00\x01\x02"
    uri = s3_store.put(content, ["embed", "corpusA", "doc2"], suffix=".bin")
    assert s3_store.get(uri) == content


def test_identical_content_maps_to_identical_key(s3_store):
    content = b"same bytes twice"
    uri1 = s3_store.put(content, ["extract", "corpusA", "docX"], suffix=".json")
    uri2 = s3_store.put(content, ["extract", "corpusA", "docX"], suffix=".json")
    assert uri1 == uri2


def test_different_content_maps_to_different_key(s3_store):
    uri1 = s3_store.put(b"content one", ["extract", "corpusA", "docY"], suffix=".json")
    uri2 = s3_store.put(b"content two", ["extract", "corpusA", "docY"], suffix=".json")
    assert uri1 != uri2


def test_put_is_idempotent_skip_on_existing_key(s3_store, monkeypatch):
    content = b"idempotent bytes"
    uri1 = s3_store.put(content, ["indexer", "corpusA", "docZ"], suffix=".json")

    calls = []
    original_put_object = s3_store._client.put_object

    def _tracking_put_object(**kwargs):
        calls.append(kwargs)
        return original_put_object(**kwargs)

    monkeypatch.setattr(s3_store._client, "put_object", _tracking_put_object)

    uri2 = s3_store.put(content, ["indexer", "corpusA", "docZ"], suffix=".json")

    assert uri2 == uri1
    assert calls == []  # put_object was never called; head_object short-circuited


def test_get_rejects_uri_from_a_different_bucket(s3_store):
    with pytest.raises(ValueError, match="bucket"):
        s3_store.get("s3://some-other-bucket/key/does/not/matter")


def test_get_rejects_non_s3_uri(s3_store):
    with pytest.raises(ValueError, match="s3://"):
        s3_store.get("file:///tmp/whatever")


def test_local_and_s3_key_layout_match(tmp_path, s3_store):
    """The two backends must agree on the key-shape contract (same key_parts,
    same suffix -> same relative path segments), so swapping the backend
    doesn't require touching handler code that constructs key_parts."""
    content = b"shared key layout"
    key_parts = ["extract", "corpusA", "docShared"]
    suffix = ".json"

    local_store = LocalArtifactStore(root=tmp_path / "artifacts")
    local_uri = local_store.put(content, key_parts, suffix=suffix)
    s3_uri = s3_store.put(content, key_parts, suffix=suffix)

    sha256 = hashlib.sha256(content).hexdigest()
    expected_relative = f"extract/corpusA/docShared/{sha256[:2]}/{sha256}{suffix}"
    assert local_uri.endswith(expected_relative)
    assert s3_uri.endswith(expected_relative)


def test_from_env_missing_vars_raises(monkeypatch):
    monkeypatch.delenv("PIPELINE_V2_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("PIPELINE_V2_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PIPELINE_V2_S3_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing required env var"):
        S3ArtifactStore.from_env()


def test_artifact_store_from_env_defaults_to_local(monkeypatch, tmp_path):
    monkeypatch.delenv("PIPELINE_V2_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("PIPELINE_V2_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    store = artifact_store_from_env()
    assert isinstance(store, LocalArtifactStore)


def test_artifact_store_from_env_s3_missing_config_raises(monkeypatch):
    monkeypatch.setenv("PIPELINE_V2_ARTIFACT_BACKEND", "s3")
    monkeypatch.delenv("PIPELINE_V2_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("PIPELINE_V2_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PIPELINE_V2_S3_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing required env var"):
        artifact_store_from_env()


def test_artifact_store_from_env_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("PIPELINE_V2_ARTIFACT_BACKEND", "azure")
    with pytest.raises(ValueError, match="not recognized"):
        artifact_store_from_env()


# --- Live SeaweedFS integration -------------------------------------------------


@pytest.fixture(scope="module")
def seaweedfs_s3_endpoint():
    """Port-forward the live in-cluster seaweedfs-s3 svc for the duration of this module.

    Bounded lifetime: the port-forward subprocess is killed in the fixture
    teardown regardless of test outcome. Skips (rather than errors) if
    kubectl/the cluster isn't reachable, so the offline suite is unaffected.
    """
    if not os.environ.get("RUN_INTEGRATION"):
        pytest.skip("integration test (set RUN_INTEGRATION=1 to run)")

    kubectl = os.path.expanduser("~/.local/bin/kubectl")
    if not os.path.exists(kubectl):
        pytest.skip(f"kubectl not found at {kubectl}")

    port = 18333
    proc = subprocess.Popen(
        [kubectl, "port-forward", "-n", "storage", "svc/seaweedfs-s3", f"{port}:8333"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20.0
        endpoint = f"http://127.0.0.1:{port}"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.skip(f"kubectl port-forward exited early: {out[:500]}")
            try:
                import httpx

                httpx.get(endpoint, timeout=1.0)
                break
            except Exception as exc:  # noqa: BLE001 - retry until timeout
                last_err = exc
                time.sleep(0.5)
        else:
            proc.kill()
            pytest.skip(f"seaweedfs-s3 port-forward never became reachable: {last_err}")
        yield endpoint
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.integration
def test_s3_artifact_store_live_seaweedfs_roundtrip(seaweedfs_s3_endpoint):
    access_key = os.environ.get("PIPELINE_V2_S3_ACCESS_KEY")
    secret_key = os.environ.get("PIPELINE_V2_S3_SECRET_KEY")
    if not access_key or not secret_key:
        pytest.skip(
            "PIPELINE_V2_S3_ACCESS_KEY/PIPELINE_V2_S3_SECRET_KEY not set "
            "(decode from `kubectl get secret -n storage seaweedfs-s3-config`)"
        )

    bucket = f"pipeline-v2-it-{int(time.time())}"
    store = S3ArtifactStore(
        endpoint_url=seaweedfs_s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
    )

    content = b"live seaweedfs round trip"
    uri = store.put(content, ["extract", "it-corpus", "it-doc"], suffix=".json")
    assert uri.startswith(f"s3://{bucket}/")

    fetched = store.get(uri)
    assert fetched == content

    # list round-trip: the object we just wrote shows up under the bucket.
    listing = store._client.list_objects_v2(Bucket=bucket)
    keys = [obj["Key"] for obj in listing.get("Contents", [])]
    assert any(key.endswith(".json") for key in keys)

    # idempotent put: re-put of identical content resolves to the same uri.
    uri2 = store.put(content, ["extract", "it-corpus", "it-doc"], suffix=".json")
    assert uri2 == uri
