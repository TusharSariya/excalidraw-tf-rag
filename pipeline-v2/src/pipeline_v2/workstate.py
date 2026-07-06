"""Work-state ledger: one row per (doc_id, corpus, stage), the sole record of
whether a pipeline stage succeeded for a document.

This exists to structurally prevent the v1 bug that destroyed 2,266 docs: a
``degraded`` outcome (partial success — e.g. OCR fallback used, or a page
truncated) was written to the same place as a real success and later read
back as "done", so it got indexed like a clean success. Two invariants close
that off for good:

  * ``state`` is one of exactly ``ok`` / ``degraded`` / ``failed`` — enforced
    by a DB ``CHECK`` constraint on the Postgres backend and by
    :class:`WorkStateError` on the in-memory backend, so a typo'd or
    made-up state can never be written.
  * The *only* sanctioned "is this indexable" read path is
    :meth:`WorkStateRepository.list_indexable`, which hard-filters
    ``state = 'ok'`` (backed by the ``indexable_work_state`` view on
    Postgres). Nothing in this module exposes a way to ask "does a row
    exist" without also asking "and is it ok" — existence alone must never
    be read as done.

Two backends implement the same :class:`WorkStateRepository` contract:

  * :class:`InMemoryWorkStateRepository` — pure Python, no I/O. Used by the
    offline unit test suite and for local dev/dry-runs.
  * :class:`PostgresWorkStateRepository` — the production backend, backed by
    ``schema.sql``. Exercised by the ``@pytest.mark.integration`` suite
    against a real ``postgres:16`` container.

Both raise :class:`WorkStateError` for the same failure modes (invalid
state, etc.) so tests can be written once against the interface and run
against either backend.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ValidationError

from pipeline_v2 import db

State = Literal["ok", "degraded", "failed"]
_VALID_STATES: frozenset[str] = frozenset({"ok", "degraded", "failed"})
_DEAD_LETTER_STATES: frozenset[str] = frozenset({"degraded", "failed"})


class WorkStateError(Exception):
    """Raised for constraint violations — invalid state, bad row shape, etc.

    Raised uniformly by both backends: the in-memory repo raises it directly,
    the Postgres repo translates ``psycopg.errors.CheckViolation`` (and other
    integrity errors) into it, so contract tests can assert on one exception
    type regardless of backend.
    """


class WorkStateRow(BaseModel):
    doc_id: str
    corpus: str
    content_sha256: str
    stage: str
    stage_fingerprint: str
    attempt: int = 1
    state: State
    reason: str | None = None
    artifact_uri: str | None = None
    run_id: str | None = None
    updated_at: datetime


@dataclass(frozen=True)
class UpsertResult:
    """Outcome of :meth:`WorkStateRepository.upsert_outcome`.

    ``changed`` is False exactly when the call was a true no-op: an existing
    row already has the same ``content_sha256`` + ``stage_fingerprint`` and
    state ``'ok'``, and the caller is reporting the same outcome again. In
    that case the row (including ``attempt`` and ``updated_at``) is left
    untouched — re-running an already-successful, unchanged stage must not
    churn the ledger or look like a new attempt.
    """

    changed: bool
    row: WorkStateRow


def _validate_state(state: str) -> None:
    if state not in _VALID_STATES:
        raise WorkStateError(f"invalid state {state!r}: must be one of {sorted(_VALID_STATES)}")


class WorkStateRepository(ABC):
    """Contract shared by the in-memory and Postgres work-state backends."""

    @abstractmethod
    def init_db(self) -> None:
        """Create/upgrade schema. No-op safe to call repeatedly."""

    @abstractmethod
    def upsert_outcome(
        self,
        *,
        doc_id: str,
        corpus: str,
        stage: str,
        content_sha256: str,
        stage_fingerprint: str,
        state: State,
        reason: str | None = None,
        artifact_uri: str | None = None,
        run_id: str | None = None,
    ) -> UpsertResult:
        """Record the outcome of one attempt at ``stage`` for ``doc_id``.

        Idempotent: if the current row already has the same
        ``content_sha256`` + ``stage_fingerprint`` and ``state='ok'``, and
        the new outcome is also ``state='ok'`` with the same sha/fingerprint,
        this is a no-op (``UpsertResult.changed is False``). Otherwise a new
        attempt is recorded (``attempt`` incremented from the prior row, or
        1 for a first-ever row).
        """

    @abstractmethod
    def get(self, doc_id: str, corpus: str, stage: str) -> WorkStateRow | None: ...

    @abstractmethod
    def get_dead_letters(
        self, stage: str | None = None, corpus: str | None = None
    ) -> list[WorkStateRow]:
        """Rows with state in ('degraded', 'failed'), optionally filtered."""

    @abstractmethod
    def list_indexable(self, corpus: str | None = None) -> list[WorkStateRow]:
        """Rows with state='ok' — the only rows ever safe to index. See module docstring."""

    @abstractmethod
    def retry_dead_letter(self, stage: str, corpus: str | None = None) -> int:
        """Reset dead-letter rows (state in ('degraded','failed')) for a fresh retry pass.

        ``state`` is deliberately left as-is (still 'degraded'/'failed', so
        it is structurally never indexable) — there is no 'pending' state in
        the schema, by design (see module docstring: no non-terminal state
        can ever be mistaken for done). "Pending-retry" instead means:
        ``attempt`` is bumped and the stale ``reason``/``artifact_uri``/
        ``run_id`` from the last failure are cleared, so the row reads as
        "eligible for reprocessing" rather than "here is why it failed last
        time". :meth:`get_dead_letters` / :meth:`claim_retry` already select
        on state alone, so a worker picks these rows back up on the next
        pass without any new state value.

        Returns the number of rows touched. Rows with state='ok' are never
        touched (retry only ever applies to dead letters).
        """

    @abstractmethod
    @contextlib.contextmanager
    def claim_retry(
        self, stage: str, corpus: str | None = None, limit: int = 1
    ) -> Iterator[list[WorkStateRow]]:
        """Exclusively claim up to ``limit`` dead-letter rows for reprocessing.

        On the Postgres backend this opens a transaction, runs
        ``SELECT ... FOR UPDATE SKIP LOCKED``, yields the claimed rows, and
        commits (releasing the row locks) when the ``with`` block exits
        normally — or rolls back on exception. A second concurrent caller
        racing the same stage/corpus will skip whatever rows the first
        caller is still holding and only see the remainder, so the same
        dead letter is never handed to two workers at once.
        """


class InMemoryWorkStateRepository(WorkStateRepository):
    """Pure-Python backend for offline unit tests and local dry-runs.

    Enforces the same invariants as the Postgres schema (state CHECK, only
    'ok' is indexable) so contract tests can run against this backend without
    a database and still catch real bugs.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], WorkStateRow] = {}

    def init_db(self) -> None:
        return None

    def upsert_outcome(
        self,
        *,
        doc_id: str,
        corpus: str,
        stage: str,
        content_sha256: str,
        stage_fingerprint: str,
        state: State,
        reason: str | None = None,
        artifact_uri: str | None = None,
        run_id: str | None = None,
    ) -> UpsertResult:
        _validate_state(state)
        key = (doc_id, corpus, stage)
        existing = self._rows.get(key)

        if (
            existing is not None
            and existing.state == "ok"
            and existing.content_sha256 == content_sha256
            and existing.stage_fingerprint == stage_fingerprint
            and state == "ok"
        ):
            return UpsertResult(changed=False, row=existing)

        try:
            row = WorkStateRow(
                doc_id=doc_id,
                corpus=corpus,
                stage=stage,
                content_sha256=content_sha256,
                stage_fingerprint=stage_fingerprint,
                attempt=(existing.attempt + 1) if existing is not None else 1,
                state=state,
                reason=reason,
                artifact_uri=artifact_uri,
                run_id=run_id,
                updated_at=datetime.now(UTC),
            )
        except ValidationError as exc:  # pragma: no cover - defensive
            raise WorkStateError(str(exc)) from exc
        self._rows[key] = row
        return UpsertResult(changed=True, row=row)

    def get(self, doc_id: str, corpus: str, stage: str) -> WorkStateRow | None:
        return self._rows.get((doc_id, corpus, stage))

    def get_dead_letters(
        self, stage: str | None = None, corpus: str | None = None
    ) -> list[WorkStateRow]:
        return sorted(
            (
                row
                for row in self._rows.values()
                if row.state in _DEAD_LETTER_STATES
                and (stage is None or row.stage == stage)
                and (corpus is None or row.corpus == corpus)
            ),
            key=lambda r: (r.corpus, r.stage, r.doc_id),
        )

    def list_indexable(self, corpus: str | None = None) -> list[WorkStateRow]:
        return sorted(
            (
                row
                for row in self._rows.values()
                if row.state == "ok" and (corpus is None or row.corpus == corpus)
            ),
            key=lambda r: (r.corpus, r.stage, r.doc_id),
        )

    def retry_dead_letter(self, stage: str, corpus: str | None = None) -> int:
        touched = 0
        for key, row in list(self._rows.items()):
            if row.stage != stage or row.state not in _DEAD_LETTER_STATES:
                continue
            if corpus is not None and row.corpus != corpus:
                continue
            self._rows[key] = row.model_copy(
                update={
                    "attempt": row.attempt + 1,
                    "reason": None,
                    "artifact_uri": None,
                    "run_id": None,
                    "updated_at": datetime.now(UTC),
                }
            )
            touched += 1
        return touched

    @contextlib.contextmanager
    def claim_retry(
        self, stage: str, corpus: str | None = None, limit: int = 1
    ) -> Iterator[list[WorkStateRow]]:
        # No real concurrency in-process; return the same rows a Postgres
        # SKIP LOCKED query would surface for a single caller.
        claimed = self.get_dead_letters(stage=stage, corpus=corpus)[:limit]
        yield claimed


class PostgresWorkStateRepository(WorkStateRepository):
    """Production backend, backed by the ``work_state`` table in schema.sql."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = db.resolve_dsn(dsn)

    def init_db(self) -> None:
        db.init_schema(self._dsn)

    @staticmethod
    def _row_from_record(record: dict) -> WorkStateRow:
        return WorkStateRow.model_validate(record)

    def upsert_outcome(
        self,
        *,
        doc_id: str,
        corpus: str,
        stage: str,
        content_sha256: str,
        stage_fingerprint: str,
        state: State,
        reason: str | None = None,
        artifact_uri: str | None = None,
        run_id: str | None = None,
    ) -> UpsertResult:
        _validate_state(state)
        with db.connect(self._dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM work_state WHERE doc_id=%s AND corpus=%s AND stage=%s "
                    "FOR UPDATE",
                    (doc_id, corpus, stage),
                )
                existing_record = cur.fetchone()
                existing = self._row_from_record(existing_record) if existing_record else None

                if (
                    existing is not None
                    and existing.state == "ok"
                    and existing.content_sha256 == content_sha256
                    and existing.stage_fingerprint == stage_fingerprint
                    and state == "ok"
                ):
                    conn.commit()
                    return UpsertResult(changed=False, row=existing)

                next_attempt = (existing.attempt + 1) if existing is not None else 1
                try:
                    cur.execute(
                        """
                        INSERT INTO work_state
                            (doc_id, corpus, content_sha256, stage, stage_fingerprint,
                             attempt, state, reason, artifact_uri, run_id, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (doc_id, corpus, stage) DO UPDATE SET
                            content_sha256 = EXCLUDED.content_sha256,
                            stage_fingerprint = EXCLUDED.stage_fingerprint,
                            attempt = EXCLUDED.attempt,
                            state = EXCLUDED.state,
                            reason = EXCLUDED.reason,
                            artifact_uri = EXCLUDED.artifact_uri,
                            run_id = EXCLUDED.run_id,
                            updated_at = now()
                        RETURNING *
                        """,
                        (
                            doc_id,
                            corpus,
                            content_sha256,
                            stage,
                            stage_fingerprint,
                            next_attempt,
                            state,
                            reason,
                            artifact_uri,
                            run_id,
                        ),
                    )
                except psycopg.errors.CheckViolation as exc:
                    conn.rollback()
                    raise WorkStateError(str(exc)) from exc
                row = self._row_from_record(cur.fetchone())
                conn.commit()
                return UpsertResult(changed=True, row=row)

    def get(self, doc_id: str, corpus: str, stage: str) -> WorkStateRow | None:
        with db.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM work_state WHERE doc_id=%s AND corpus=%s AND stage=%s",
                    (doc_id, corpus, stage),
                )
                record = cur.fetchone()
                return self._row_from_record(record) if record else None

    def get_dead_letters(
        self, stage: str | None = None, corpus: str | None = None
    ) -> list[WorkStateRow]:
        clauses = ["state IN ('degraded', 'failed')"]
        params: list[str] = []
        if stage is not None:
            clauses.append("stage = %s")
            params.append(stage)
        if corpus is not None:
            clauses.append("corpus = %s")
            params.append(corpus)
        query = (
            "SELECT * FROM work_state WHERE "
            + " AND ".join(clauses)
            + " ORDER BY corpus, stage, doc_id"
        )
        with db.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return [self._row_from_record(r) for r in cur.fetchall()]

    def list_indexable(self, corpus: str | None = None) -> list[WorkStateRow]:
        query = "SELECT * FROM indexable_work_state"
        params: list[str] = []
        if corpus is not None:
            query += " WHERE corpus = %s"
            params.append(corpus)
        query += " ORDER BY corpus, stage, doc_id"
        with db.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return [self._row_from_record(r) for r in cur.fetchall()]

    def retry_dead_letter(self, stage: str, corpus: str | None = None) -> int:
        clauses = ["stage = %s", "state IN ('degraded', 'failed')"]
        params: list[str] = [stage]
        if corpus is not None:
            clauses.append("corpus = %s")
            params.append(corpus)
        query = (
            "UPDATE work_state SET attempt = attempt + 1, reason = NULL, "
            "artifact_uri = NULL, run_id = NULL, updated_at = now() WHERE " + " AND ".join(clauses)
        )
        with db.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                touched = cur.rowcount
            conn.commit()
        return touched

    @contextlib.contextmanager
    def claim_retry(
        self, stage: str, corpus: str | None = None, limit: int = 1
    ) -> Iterator[list[WorkStateRow]]:
        clauses = ["stage = %s", "state IN ('degraded', 'failed')"]
        params: list[str] = [stage]
        if corpus is not None:
            clauses.append("corpus = %s")
            params.append(corpus)
        query = (
            "SELECT * FROM work_state WHERE "
            + " AND ".join(clauses)
            + " ORDER BY corpus, stage, doc_id LIMIT %s FOR UPDATE SKIP LOCKED"
        )
        params.append(limit)

        conn = db.connect(self._dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                claimed = [self._row_from_record(r) for r in cur.fetchall()]
            yield claimed
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()
