"""harvest-work: drain ONE source's queue.

A single-threaded pull loop: read a message (which hides it for a visibility
timeout), resolve the seed's external identity, fetch the source, persist the
result, then archive the message (the ack) — and exit when the queue is empty
(drain-then-exit). Messages are read through the Supabase `pgmq_public` wrappers
over the REST client.

Ack discipline (pgmq is at-least-once):
  * success             -> write, then archive
  * source not found    -> mark_not_found, then archive (terminal, don't retry)
  * unexpected failure  -> leave unarchived; pgmq redelivers after the VT
  * too many redelivers -> mark_error, then archive (poison-pill backstop)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import contracts
from .persistence.base import Write
from .persistence.models import Seed, s2_write
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


SOURCES: dict[str, Source] = {
    "arxiv": Source(
        "harvest_arxiv",
        lambda _: ArxivSource(),
        identify=[lambda seed: seed.arxiv_id],
        build_write=s2_write,
    ),
    "openalex": Source(
        "harvest_openalex",
        lambda _: OpenAlexSource(),
        identify=[lambda seed: seed.openalex_id],
        build_write=s2_write,
    ),
    "s2": Source(
        "harvest_s2",
        lambda _: SemanticScholarSource(),
        # S2-namespaced external ids, tried in order: arXiv first, DOI as fallback.
        identify=[
            lambda seed: f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else None,
            lambda seed: f"DOI:{seed.doi}" if seed.doi else None,
        ],
        build_write=s2_write,
    ),
}

# Visibility timeout: how long a read message stays hidden while we work on it.
# If we don't archive within this window, pgmq redelivers it (at-least-once).
VT_SECONDS = 120

# Redelivery cap: a message that keeps failing (read_ct climbing) is treated as a
# poison pill past this many attempts — marked error and archived, not retried.
MAX_READ_CT = 5


def _archive(q: Any, queue: str, msg_id: int) -> None:
    q.rpc("archive", {"queue_name": queue, "message_id": msg_id}).execute()


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = WorkerSettings()
    source_key = settings.source
    try:
        source = SOURCES[source_key]
    except KeyError:
        raise SystemExit(f"--source must be one of {sorted(SOURCES)}, got {source_key!r}") from None

    queue = source.queue
    identify = source.identify
    build_write = source.build_write
    paper_source = source.factory(settings)
    store = SupabasePersistence.from_settings(SupabaseSettings())
    q = store.supabase.schema("pgmq_public")

    logger.info("draining queue %s", queue)
    drained = 0
    while True:
        resp = q.rpc("read", {"queue_name": queue, "sleep_seconds": VT_SECONDS, "n": 10}).execute()
        rows = resp.data
        if not rows:
            break

        rec = rows[0]
        event = contracts.parse(rec["message"])
        paper_id = event.paper_id
        read_ct = rec.get("read_ct") or 0
        logger.info("read msg_id=%s paper_id=%s read_ct=%s", rec["msg_id"], paper_id, read_ct)

        # Poison-pill backstop: stop redelivering a message we keep failing on.
        if read_ct > MAX_READ_CT:
            logger.error("paper_id=%s exceeded %d attempts — marking error", paper_id, MAX_READ_CT)
            store.mark_error(source_key, paper_id, f"exceeded {MAX_READ_CT} attempts")
            _archive(q, queue, rec["msg_id"])
            continue

        try:
            seed = store.load_seed(paper_id)
            # Try each candidate id in order (e.g. s2: arXiv, then DOI), stopping
            # at the first that S2 actually resolves. A transient fetch error
            # propagates out of get_paper to the redelivery path below; only a
            # genuine not-found returns None and lets us fall through to the next
            # id. mark_not_found fires only when every candidate is not-found.
            record = None
            tried: list[str] = []
            for pick in identify:
                identifier = pick(seed)
                if not identifier:
                    continue
                tried.append(identifier)
                record = paper_source.get_paper(identifier)
                if record is not None:
                    break

            if record is None:
                logger.info(
                    "no %s paper for seed %s (tried %s)",
                    source_key,
                    paper_id,
                    tried or "no ids",
                )
                store.mark_not_found(source_key, paper_id)
                _archive(q, queue, rec["msg_id"])
                continue

            store.write(build_write(paper_id, record))
            _archive(q, queue, rec["msg_id"])  # ack only after a successful write
            drained += 1
            logger.info("persisted seed %s (id=%s)", paper_id, identifier)
        except Exception:
            # Leave unarchived: pgmq redelivers after the VT. The read_ct cap
            # above turns a permanently-bad seed into a terminal error instead of
            # an infinite loop.
            logger.exception("paper_id=%s failed — leaving for redelivery", paper_id)

    logger.info("queue %s empty — drained %d, exiting", queue, drained)
