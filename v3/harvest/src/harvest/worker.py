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
from .persistence.models import (
    Seed,
    acl_anthology_write,
    arxiv_write,
    core_write,
    crossref_write,
    datacite_write,
    dblp_write,
    figshare_write,
    internet_archive_write,
    iter_s2_writes,
    openalex_write,
    opencitations_write,
    openreview_write,
    osf_write,
    unpaywall_write,
    zenodo_write,
)
from .persistence.supabase import SupabasePersistence
from .settings import HarvestSettings, SupabaseSettings, WorkerSettings
from .sources.acl_anthology import AclAnthologySource
from .sources.arxiv import ArxivSource
from .sources.base import PaperSource
from .sources.core import CoreSource
from .sources.crossref import CrossrefSource
from .sources.datacite import DataCiteSource
from .sources.dblp import DblpSource
from .sources.figshare import FigshareSource
from .sources.internet_archive import InternetArchiveSource
from .sources.openalex import OpenAlexSource
from .sources.opencitations import OpenCitationsSource
from .sources.openreview import OpenReviewSource
from .sources.osf import OsfSource
from .sources.semantic_scholar import SemanticScholarSource
from .sources.unpaywall import UnpaywallSource
from .sources.zenodo import ZenodoSource

logger = logging.getLogger("harvest.worker")


@dataclass(frozen=True)
class Source:
    """Everything the worker needs to drain one source's queue.

    ``identify`` (pick the fetch id from a seed) and ``build_write`` (turn a
    fetched record into a Write, or a list of Writes for chunked sources like
    S2) are None until a source's record->Write mapper exists — such a source
    can be seeded but not yet drained.
    """

    queue: str
    factory: Callable[[HarvestSettings], PaperSource]
    identify: list[Callable[[Seed], str | None]]
    build_write: Callable[[int, Any], Write | list[Write]]
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
        # DOI path is one Works()[doi] call per id (W… ids batch). batch_size=100
        # + VT=120 let a 100-DOI read overrun visibility → stampede redelivery.
        batch_size=10,
    ),
    "s2": Source(
        "harvest_s2",
        lambda _: SemanticScholarSource(),
        # S2-namespaced external id: the first present of arXiv, then DOI.
        identify=[
            lambda seed: f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else None,
            lambda seed: f"DOI:{seed.doi}" if seed.doi else None,
        ],
        # build_write unused for s2 — worker streams via iter_s2_writes +
        # SemanticScholarSource.iter_related_chunks (see main loop).
        build_write=lambda _pid, _rec: [],
        # One id per request: subject metadata + streamed cite/ref pages.
        batch_size=1,
    ),
    "unpaywall": Source(
        "harvest_unpaywall",
        lambda s: UnpaywallSource(s.unpaywall_email),
        identify=[lambda seed: seed.doi],
        build_write=unpaywall_write,
        batch_size=50,
    ),
    "crossref": Source(
        "harvest_crossref",
        lambda s: CrossrefSource(s.crossref_mailto),
        identify=[lambda seed: seed.doi],
        build_write=crossref_write,
        batch_size=20,
    ),
    "dblp": Source(
        "harvest_dblp",
        lambda _: DblpSource(),
        identify=[lambda seed: seed.dblp_id],
        build_write=dblp_write,
        batch_size=1,
    ),
    "openreview": Source(
        "harvest_openreview",
        lambda _: OpenReviewSource(),
        identify=[lambda seed: seed.openreview_id],
        build_write=openreview_write,
        batch_size=100,
    ),
    "acl_anthology": Source(
        "harvest_acl_anthology",
        lambda _: AclAnthologySource(),
        identify=[lambda seed: seed.anthology_id],
        build_write=acl_anthology_write,
        batch_size=500,
    ),
    "opencitations": Source(
        "harvest_opencitations",
        lambda _: OpenCitationsSource(),
        identify=[lambda seed: seed.doi],
        build_write=opencitations_write,
        # batch_size=1: get_papers fetches serially and a single heavily-cited DOI
        # can stall past the read timeout (a well-cited paper the API can't serve).
        # At 1, that slow DOI poisons only its own message — it can't take down a
        # batch of faster neighbors (which would leave the whole batch unarchived).
        batch_size=1,
    ),
    "datacite": Source(
        "harvest_datacite",
        lambda _: DataCiteSource(),
        identify=[lambda seed: seed.doi],
        build_write=datacite_write,
        batch_size=50,
    ),
    "core": Source(
        "harvest_core",
        lambda s: CoreSource(s.core_api_key),
        identify=[lambda seed: seed.doi],
        build_write=core_write,
        batch_size=1,
    ),
    "internet_archive": Source(
        "harvest_internet_archive",
        lambda _: InternetArchiveSource(),
        identify=[lambda seed: seed.ia_identifier],
        build_write=internet_archive_write,
        batch_size=25,
    ),
    "zenodo": Source(
        "harvest_zenodo",
        lambda _: ZenodoSource(),
        identify=[lambda seed: seed.doi],
        build_write=zenodo_write,
        # Serial DOI lookups; a mid-batch 429 aborts the whole read. Keep small
        # so one rate-limit doesn't strand dozens of already-resolved not_founds.
        batch_size=10,
    ),
    "figshare": Source(
        "harvest_figshare",
        lambda _: FigshareSource(),
        identify=[lambda seed: seed.doi],
        build_write=figshare_write,
        batch_size=10,
    ),
    "osf": Source(
        "harvest_osf",
        lambda _: OsfSource(),
        identify=[lambda seed: seed.doi],
        build_write=osf_write,
        batch_size=25,
    ),
}

# Visibility timeout: how long a read message stays hidden while we work on it.
# If we don't archive within this window, pgmq redelivers it (at-least-once).
# S2 paginates mega-cite graphs (10k+) with streamed upserts — needs a long VT.
# OpenAlex DOI path is serial + retries; 10 ids comfortably fit under 600s.
VT_SECONDS = 120
S2_VT_SECONDS = 900
OPENALEX_VT_SECONDS = 600


def _vt_seconds(source_key: str) -> int:
    if source_key == "s2":
        return S2_VT_SECONDS
    if source_key == "openalex":
        return OPENALEX_VT_SECONDS
    return VT_SECONDS


# How long to wait for a full batch to accumulate before reading whatever is there.
WAIT_SECONDS = 120.0

# Continuous empty-queue wait: stop polling and read (usually []) so drain-then-exit
# workers die instead of hammering PostgREST. KEDA re-spawns within ~15s when work
# reappears.
EMPTY_EXIT_SECONDS = 10.0

# Visible-length poll pacing: fast for the first second, then 1s.
POLL_FAST_SECONDS = 0.1
POLL_FAST_FOR_SECONDS = 1.0
POLL_SLOW_SECONDS = 1.0

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
            vt_seconds=_vt_seconds(source_key),
            wait_seconds=WAIT_SECONDS,
            empty_exit_seconds=EMPTY_EXIT_SECONDS,
            poll_fast_seconds=POLL_FAST_SECONDS,
            poll_fast_for_seconds=POLL_FAST_FOR_SECONDS,
            poll_slow_seconds=POLL_SLOW_SECONDS,
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

                if source_key == "s2":
                    assert isinstance(paper_source, SemanticScholarSource)
                    assert record.paper_id is not None
                    writes = iter_s2_writes(
                        paper_id,
                        record,
                        citation_chunks=paper_source.iter_related_chunks(
                            "citations",
                            record.paper_id,
                        ),
                        reference_chunks=paper_source.iter_related_chunks(
                            "references",
                            record.paper_id,
                        ),
                    )
                else:
                    payload = build_write(paper_id, record)
                    writes = payload if isinstance(payload, list) else [payload]
                for w in writes:
                    store.write(w)
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
