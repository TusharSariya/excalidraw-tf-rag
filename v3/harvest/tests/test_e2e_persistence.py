"""End-to-end: a mock S2 API response all the way to the DB write calls.

Path exercised: raw API JSON -> PaperRecord.from_dict -> from_paper_record ->
S2GraphWrite.to_write -> SupabasePersistence.write -> PostgREST upserts.

The DB is a fake that records every ``table(...).upsert(...).execute()`` so the
test asserts *what would be persisted* — the tables, the upsert conflict keys,
the FK-safe ordering, and a JSON-serializable payload — without a live Supabase.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from harvest.persistence.models.s2 import SPECTER_V1_DIM, from_paper_record, s2_write
from harvest.persistence.supabase import SupabasePersistence
from harvest.sources.models.s2 import Embedding, PaperRecord


class _FakeTable:
    """One `table(name)` handle; records the upsert when `.execute()` fires."""

    def __init__(self, name: str, calls: list[dict[str, Any]]) -> None:
        self._name = name
        self._calls = calls
        self._op: dict[str, Any] | None = None

    def upsert(self, rows: list[dict], *, on_conflict: str | None = None) -> _FakeTable:
        self._op = {"table": self._name, "rows": rows, "on_conflict": on_conflict}
        return self

    def execute(self) -> Any:
        assert self._op is not None, "execute() before an operation"
        self._calls.append(self._op)
        return type("Resp", (), {"data": self._op["rows"]})()


class FakeSupabase:
    """Minimal stand-in for the supabase-py Client: records write ops in order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(name, self.calls)


def test_e2e_s2_response_to_persistence(s2_paper_forceatlas2: dict) -> None:
    # 1. Mock API response -> typed record. Fixture vector is truncated, so drop
    #    the embedding (the 768-d guard is covered separately below).
    rec = replace(PaperRecord.from_dict(s2_paper_forceatlas2), embedding=None)

    # 2. record -> mapper -> FK-ordered write payload.
    write = from_paper_record(rec).to_write()

    # 3. write payload -> backend -> recorded DB ops.
    client = FakeSupabase()
    SupabasePersistence(client).write(write)

    calls = client.calls
    order = [c["table"] for c in calls]
    by_table = {c["table"]: c for c in calls}

    # Every graph table with data was written; none written empty.
    assert set(order) == {
        "s2_paper",
        "s2_author",
        "s2_paper_author",
        "s2_reference",
        "s2_citation",
    }
    assert len(order) == len(set(order))  # each table upserted exactly once
    assert "s2_embedding" not in order  # embedding was None -> group dropped

    # FK-safe ordering: parents before the rows that reference them.
    assert order.index("s2_paper") < order.index("s2_paper_author")
    assert order.index("s2_author") < order.index("s2_paper_author")
    assert order.index("s2_paper") < order.index("s2_reference")
    assert order.index("s2_paper") < order.index("s2_citation")

    # Conflict keys are the tables' real PKs (upsert idempotency).
    assert by_table["s2_paper"]["on_conflict"] == "paper_id"
    assert by_table["s2_author"]["on_conflict"] == "author_id"
    assert by_table["s2_paper_author"]["on_conflict"] == "paper_id,author_id"
    assert by_table["s2_reference"]["on_conflict"] == "paper_id,referenced_paper_id"
    assert by_table["s2_citation"]["on_conflict"] == "paper_id,citing_paper_id"

    # Subject paper survives the whole pipeline with its content intact.
    subject = next(r for r in by_table["s2_paper"]["rows"] if r["paper_id"] == rec.paper_id)
    assert subject["title"] == rec.title
    assert "__table__" not in subject  # ClassVar metadata never becomes a column
    assert "__conflict__" not in subject
    assert isinstance(subject["fetched_at"], str)  # datetime -> ISO str (mode="json")

    # Edges point back at the subject and land in the paper set (no dangling FKs).
    paper_ids = {r["paper_id"] for r in by_table["s2_paper"]["rows"]}
    for edge in by_table["s2_reference"]["rows"]:
        assert edge["paper_id"] == rec.paper_id
        assert edge["referenced_paper_id"] in paper_ids
    for edge in by_table["s2_citation"]["rows"]:
        assert edge["paper_id"] == rec.paper_id
        assert edge["citing_paper_id"] in paper_ids

    # The entire payload is JSON-serializable — the real client would send JSON.
    json.dumps(calls)


def test_e2e_worker_payload_writes_graph_and_outcome(s2_paper_forceatlas2: dict) -> None:
    # This is exactly what the worker persists per message: s2_write(seed_id, record)
    # -> the s2_* graph rows PLUS the source_semantic_scholar outcome row.
    seed_paper_id = 42  # the seed's bigint id (distinct from S2's text paper id)
    rec = replace(PaperRecord.from_dict(s2_paper_forceatlas2), embedding=None)

    client = FakeSupabase()
    SupabasePersistence(client).write(s2_write(seed_paper_id, rec))

    by_table = {c["table"]: c for c in client.calls}
    assert "source_semantic_scholar" in by_table  # outcome row written
    assert "s2_paper" in by_table  # graph rows written too

    (outcome,) = by_table["source_semantic_scholar"]["rows"]
    assert outcome["paper_id"] == seed_paper_id  # keyed by the SEED bigint...
    assert outcome["s2_paper_id"] == rec.paper_id  # ...S2's text id lives here
    assert outcome["status"] == "succeeded"
    assert outcome["title"] == rec.title
    assert by_table["source_semantic_scholar"]["on_conflict"] == "paper_id"

    # The outcome row trails the graph (it FKs only into seeds, already present).
    order = [c["table"] for c in client.calls]
    assert order.index("s2_paper") < order.index("source_semantic_scholar")


def test_e2e_includes_embedding_when_present(s2_paper_forceatlas2: dict) -> None:
    rec = replace(
        PaperRecord.from_dict(s2_paper_forceatlas2),
        embedding=Embedding(model="specter_v1", vector=[0.0] * SPECTER_V1_DIM),
    )
    write = from_paper_record(rec).to_write()

    client = FakeSupabase()
    SupabasePersistence(client).write(write)

    order = [c["table"] for c in client.calls]
    assert order[-1] == "s2_embedding"  # child of s2_paper -> written last
    emb = next(c for c in client.calls if c["table"] == "s2_embedding")
    assert emb["on_conflict"] == "paper_id"
    (row,) = emb["rows"]
    assert row["paper_id"] == rec.paper_id
    assert len(row["embedding"]) == SPECTER_V1_DIM
