"""Content-addressed artifact store abstraction.

Stage handlers (extract/embed/indexer) never write files directly; they go
through :class:`ArtifactStore` so the storage backend can move later (see
``docs/rearchitecture-v2-decision-log.md`` — SeaweedFS is the chosen
production backend) without touching handler code.

``put`` is content-addressed: the object key is derived from the caller's
``key_parts`` (stage/corpus/doc_id/stage_fingerprint — enough to make re-runs
of the same message land on the same key) plus the sha256 of the content
itself, so two calls with identical bytes always resolve to the same URI
(idempotent writes, per decision log D7 "1A: lock >= extract budget,
idempotent artifacts absorb duplicates").

Backend selection (``LocalArtifactStore`` vs ``S3ArtifactStore``) goes
through :func:`artifact_store_from_env`, keyed off ``PIPELINE_V2_ARTIFACT_BACKEND``
(``"local"`` default, or ``"s3"``); every handler's default-store wiring
(``extract.py``, ``embed.py``, ``indexer.py``) calls that one factory instead
of hardcoding ``LocalArtifactStore()``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_ARTIFACT_ROOT_ENV = "PIPELINE_V2_ARTIFACT_ROOT"
_ARTIFACT_BACKEND_ENV = "PIPELINE_V2_ARTIFACT_BACKEND"
_S3_ENDPOINT_ENV = "PIPELINE_V2_S3_ENDPOINT"
_S3_ACCESS_KEY_ENV = "PIPELINE_V2_S3_ACCESS_KEY"
_S3_SECRET_KEY_ENV = "PIPELINE_V2_S3_SECRET_KEY"
_S3_BUCKET_ENV = "PIPELINE_V2_S3_BUCKET"
_S3_DEFAULT_BUCKET = "pipeline-v2-campaign"


def _default_local_root() -> Path:
    raw = os.getenv(_ARTIFACT_ROOT_ENV, "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
    # pipeline-v2/data/artifacts, sibling to schema.sql etc.
    return Path(__file__).resolve().parents[2] / "data" / "artifacts"


def _safe_segment(value: str) -> str:
    """Keep artifact keys filesystem-safe without hiding the original value entirely."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    return cleaned[:200] or "_"


class ArtifactStore(ABC):
    """Content-addressed put/get for stage artifacts."""

    @abstractmethod
    def put(self, content: bytes, key_parts: Sequence[str], *, suffix: str = "") -> str:
        """Write ``content``, keyed by ``key_parts`` + its own sha256. Returns a URI."""

    @abstractmethod
    def get(self, uri: str) -> bytes:
        """Read back the bytes previously written at ``uri``."""


class LocalArtifactStore(ArtifactStore):
    """Local-filesystem backend. Default for now; root configurable via env.

    Layout: ``<root>/<key_parts...>/<sha256[:2]>/<sha256><suffix>``. The
    sha256-of-content prefix directory keeps any one directory from growing
    unbounded and doubles as a cheap collision check (identical content
    always maps to the identical path, so a re-run's ``put`` is a pure
    overwrite-with-identical-bytes rather than a new object).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _default_local_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, sha256: str, key_parts: Sequence[str], suffix: str) -> Path:
        segments = [_safe_segment(part) for part in key_parts]
        return self.root.joinpath(*segments, sha256[:2], f"{sha256}{suffix}")

    def put(self, content: bytes, key_parts: Sequence[str], *, suffix: str = "") -> str:
        sha256 = hashlib.sha256(content).hexdigest()
        path = self._path_for(sha256, key_parts, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp_path.write_bytes(content)
        os.replace(tmp_path, path)  # atomic within the same filesystem
        return path.resolve().as_uri()

    def get(self, uri: str) -> bytes:
        if uri.startswith("file://"):
            from urllib.parse import unquote, urlparse

            path = Path(unquote(urlparse(uri).path))
        else:
            path = Path(uri)
        return path.read_bytes()


class S3ArtifactStore(ArtifactStore):
    """S3-compatible backend, against the SeaweedFS S3 gateway (namespace ``storage``).

    Layout mirrors :class:`LocalArtifactStore`: object key is
    ``<key_parts...>/<sha256[:2]>/<sha256><suffix>`` (``/``-joined, same
    ``_safe_segment`` sanitizing) so the two backends are interchangeable —
    an artifact written under one scheme has the identical key shape under
    the other, only the storage medium changes.

    URIs are ``s3://<bucket>/<key>``. ``get`` requires the ``uri`` to name a
    bucket this store is configured against (cross-bucket reads aren't a
    supported use case here — each pipeline-v2 deployment talks to exactly
    one artifact bucket).

    ``put`` is immutable + idempotent: it HEADs the key first and skips the
    upload if the object already exists (identical content, per
    content-addressing, so this is a pure no-op re-run rather than an
    overwrite race). SeaweedFS's S3 gateway needs path-style addressing
    (``endpoint/bucket/key``, not ``bucket.endpoint/key`` virtual-hosted
    style) since it has no wildcard DNS in front of it.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str = _S3_DEFAULT_BUCKET,
        region: str = "us-east-1",
        client: object | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        if client is not None:
            self._client = client
        else:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                config=Config(s3={"addressing_style": "path"}),
            )
        self._ensure_bucket()

    @classmethod
    def from_env(cls) -> S3ArtifactStore:
        """Build from ``PIPELINE_V2_S3_*`` env vars. Raises ``RuntimeError`` if incomplete."""
        endpoint = os.getenv(_S3_ENDPOINT_ENV, "").strip()
        access_key = os.getenv(_S3_ACCESS_KEY_ENV, "").strip()
        secret_key = os.getenv(_S3_SECRET_KEY_ENV, "").strip()
        bucket = os.getenv(_S3_BUCKET_ENV, "").strip() or _S3_DEFAULT_BUCKET
        missing = [
            name
            for name, value in (
                (_S3_ENDPOINT_ENV, endpoint),
                (_S3_ACCESS_KEY_ENV, access_key),
                (_S3_SECRET_KEY_ENV, secret_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"S3ArtifactStore.from_env: missing required env var(s): {', '.join(missing)}"
            )
        return cls(
            endpoint_url=endpoint, access_key=access_key, secret_key=secret_key, bucket=bucket
        )

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in (404, 400):
                raise
            self._client.create_bucket(Bucket=self.bucket)

    def _key_for(self, sha256: str, key_parts: Sequence[str], suffix: str) -> str:
        segments = [_safe_segment(part) for part in key_parts]
        segments.extend([sha256[:2], f"{sha256}{suffix}"])
        return "/".join(segments)

    def put(self, content: bytes, key_parts: Sequence[str], *, suffix: str = "") -> str:
        from botocore.exceptions import ClientError

        sha256 = hashlib.sha256(content).hexdigest()
        key = self._key_for(sha256, key_parts, suffix)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            # Object already present; content-addressed keys mean identical
            # bytes, so this put is a no-op (immutable-write idempotency).
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status != 404:
                raise
            self._client.put_object(Bucket=self.bucket, Key=key, Body=content)
        return f"s3://{self.bucket}/{key}"

    def get(self, uri: str) -> bytes:
        bucket, key = self._parse_uri(uri)
        if bucket != self.bucket:
            raise ValueError(
                f"S3ArtifactStore configured for bucket {self.bucket!r}, "
                f"but uri names bucket {bucket!r}: {uri!r}"
            )
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"not an s3:// uri: {uri!r}")
        rest = uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"malformed s3:// uri (need bucket and key): {uri!r}")
        return bucket, key


def artifact_store_from_env() -> ArtifactStore:
    """Select an :class:`ArtifactStore` backend via ``PIPELINE_V2_ARTIFACT_BACKEND``.

    ``"local"`` (default) -> :class:`LocalArtifactStore`; ``"s3"`` ->
    :class:`S3ArtifactStore` built from ``PIPELINE_V2_S3_*`` env vars. This is
    the one place handlers should call for their default store, so the
    backend switch is a single env var rather than a per-handler code change.
    """
    backend = os.getenv(_ARTIFACT_BACKEND_ENV, "local").strip().lower()
    if backend in ("", "local"):
        return LocalArtifactStore()
    if backend == "s3":
        logger.info("artifact_store_from_env: using S3ArtifactStore")
        return S3ArtifactStore.from_env()
    raise ValueError(
        f"{_ARTIFACT_BACKEND_ENV}={backend!r} not recognized; expected 'local' or 's3'"
    )
