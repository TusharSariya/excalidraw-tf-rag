"""harvest-work: drain ONE source's queue.

A single-threaded pull loop: read a batch of messages (each hidden for a
visibility timeout), resolve each seed's external identity, batch-fetch the
source, persist results, then archive each message (the ack) — and exit when
the queue is empty (drain-then-exit). Messages are read through the Supabase
`pgmq_public` wrappers over the REST client.

Ack discipline (pgmq is at-least-once):
  * success             -> write, then archive
  * source not found    -> mark_not_found, then archive (terminal, don't retry)
  * unexpected failure  -> leave unarchived; pgmq redelivers after the VT
  * too many redelivers -> mark_error, then archive (poison-pill backstop)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import contracts
from .persistence.base import Write
from .persistence.models import Seed, arxiv_write, openalex_write, s2_write
from .persistence.supabase import SupabasePersistence
from .settings import HarvestSettings, SupabaseSettings, WorkerSettings
from .sources.arxiv import ArxivSource
from .sources.base import PaperSource
from .sources.openalex import OpenAlexSource
from .sources.semantic_scholar import SemanticScholarSource

logger = logging.getLogger("harvest.worker")


@dataclass(frozen=True)
class Source:
    """Everything the worker needs to drain one source's queue.

    ``identify`` (pick the fetch id from a seed) and ``build_write`` (turn a
    fetched record into a Write) are None until a source's record->Write mapper
    exists — such a source can be seeded but not yet drained.
    """

    queue: str
    factory: Callable[[HarvestSettings], PaperSource]
    identify: list[Callable[[Seed], str | None]]
    build_write: Callable[[int, Any], Write]
    batch_size: int = 1
    max_response_size: int = 10485760  # bytes, 10mb
    max_citations: int = 9999


@dataclass(frozen=True)
class _Pending:
    """One in-flight queue message, keyed by seed paper_id for the batch."""

    paper_id: int
    msg_id: int


SOURCES: dict[str, Source] = {
    "arxiv": Source(
        "harvest_arxiv",
        lambda _: ArxivSource(),
        identify=[lambda seed: seed.arxiv_id],
        build_write=arxiv_write,
        batch_size=100,
        max_response_size=10485760,  # bytes, 10mb
        max_citations=9999,
    ),
    "openalex": Source(
        "harvest_openalex",
        lambda s: OpenAlexSource(s.openalex_api_key),
        # OpenAlex resolves a DOI or an OpenAlex id — never an arXiv id. Use the
        # first present of openalex_id, then DOI (most seeds carry only a DOI).
        identify=[lambda seed: seed.openalex_id, lambda seed: seed.doi],
        build_write=openalex_write,
        batch_size=100,
    ),
    "s2": Source(
        "harvest_s2",
        lambda _: SemanticScholarSource(),
        # S2-namespaced external id: the first present of arXiv, then DOI.
        identify=[
            lambda seed: f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else None,
            lambda seed: f"DOI:{seed.doi}" if seed.doi else None,
        ],
        build_write=s2_write,
        batch_size=500,
    ),
}

# Visibility timeout: how long a read message stays hidden while we work on it.
# If we don't archive within this window, pgmq redelivers it (at-least-once).
VT_SECONDS = 120

# How long to wait for a full batch to accumulate before reading whatever is there.
WAIT_SECONDS = 120.0

# Redelivery cap: a message that keeps failing (read_ct climbing) is treated as a
# poison pill past this many attempts — marked error and archived, not retried.
MAX_READ_CT = 5


def _archive(q: Any, queue: str, msg_id: int) -> None:
    q.rpc("archive", {"queue_name": queue, "message_id": msg_id}).execute()


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = WorkerSettings()
    source_key = settings.source
    if source_key is None or source_key not in SOURCES:
        raise SystemExit(f"--source must be one of {sorted(SOURCES)}, got {source_key!r}")
    source = SOURCES[source_key]

    queue = source.queue
    identify = source.identify
    build_write = source.build_write
    paper_source = source.factory(settings)
    store = SupabasePersistence.from_settings(SupabaseSettings())
    q = store.supabase.schema("pgmq_public")

    logger.info("draining queue %s", queue)
    drained = 0
    while True:
        rows = store.read_batch(
            queue,
            n=source.batch_size,
            vt_seconds=VT_SECONDS,
            wait_seconds=WAIT_SECONDS,
        )
        if not rows:
            break

        # paper_id -> in-flight message. Poison pills are archived immediately
        # and never enter the fetch path.
        pending: dict[int, _Pending] = {}
        for rec in rows:
            event = contracts.parse(rec["message"])
            paper_id = event.paper_id
            msg_id = rec["msg_id"]
            read_ct = rec.get("read_ct") or 0
            logger.info("read msg_id=%s paper_id=%s read_ct=%s", msg_id, paper_id, read_ct)

            if read_ct > MAX_READ_CT:
                logger.error(
                    "paper_id=%s exceeded %d attempts — marking error",
                    paper_id,
                    MAX_READ_CT,
                )
                store.mark_error(source_key, paper_id, f"exceeded {MAX_READ_CT} attempts")
                _archive(q, queue, msg_id)
                continue

            pending[paper_id] = _Pending(paper_id=paper_id, msg_id=msg_id)

        if not pending:
            continue

        try:
            seeds = store.load_seeds(list(pending.keys()))

            # Missing seed rows are a data bug — terminal, don't spin on redelivery.
            for paper_id, item in list(pending.items()):
                if paper_id not in seeds:
                    logger.error("paper_id=%s: seed row missing — marking error", paper_id)
                    store.mark_error(source_key, paper_id, "seed row missing")
                    _archive(q, queue, item.msg_id)
                    del pending[paper_id]

            if not pending:
                continue

            # One fetch id per seed: the first usable identifier in identify
            # order (e.g. ARXIV before DOI). Taking only the first bounds the
            # deduped batch to one id per message, so it can never exceed the
            # source's per-request id cap (e.g. S2 rejects batches over 500).
            candidates: dict[int, str] = {}
            fetch_ids: list[str] = []
            seen: set[str] = set()
            for paper_id, seed in seeds.items():
                if paper_id not in pending:
                    continue
                for pick in identify:
                    identifier = pick(seed)
                    if not identifier:
                        continue
                    candidates[paper_id] = identifier
                    if identifier not in seen:
                        seen.add(identifier)
                        fetch_ids.append(identifier)
                    break

            records = paper_source.get_papers(fetch_ids) if fetch_ids else {}

            for paper_id, item in pending.items():
                identifier = candidates.get(paper_id)
                record = records.get(identifier) if identifier else None

                if record is None:
                    logger.info(
                        "no %s paper for seed %s (tried %s)",
                        source_key,
                        paper_id,
                        identifier or "no ids",
                    )
                    store.mark_not_found(source_key, paper_id)
                    _archive(q, queue, item.msg_id)
                    continue

                store.write(build_write(paper_id, record))
                _archive(q, queue, item.msg_id)
                drained += 1
                logger.info("persisted seed %s (id=%s)", paper_id, identifier)
        except Exception:
            # Leave unarchived messages for redelivery. Already-archived items in
            # this batch stay archived (writes are upsert-idempotent on retry).
            logger.exception(
                "batch failed for paper_ids=%s — leaving unarchived for redelivery",
                sorted(pending),
            )

    logger.info("queue %s empty — drained %d, exiting", queue, drained)
