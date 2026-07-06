"""Postgres connection + schema bootstrap shared by workstate and manifest_migrate.

DSN resolution mirrors the rest of the repo's env-var convention (e.g.
``RAG_GPU_GATEWAY_URL``): an explicit argument wins, then
``PIPELINE_V2_DATABASE_URL``, then a docker-compose-friendly local default.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/pipeline_v2"

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def resolve_dsn(dsn: str | None = None) -> str:
    return dsn or os.environ.get("PIPELINE_V2_DATABASE_URL", DEFAULT_DSN)


def connect(dsn: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(resolve_dsn(dsn), autocommit=autocommit)


def init_schema(dsn: str | None = None) -> None:
    """Idempotently apply schema.sql (safe to call on every process start)."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql)
