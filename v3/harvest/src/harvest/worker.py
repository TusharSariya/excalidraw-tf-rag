"""harvest-work: drain ONE source's queue.

A single-threaded pull loop: read a message (which hides it for a visibility
timeout), process it, archive it (the ack), and exit when the queue is empty
(drain-then-exit). Messages are read through the Supabase `pgmq_public` wrappers
over the REST client.
"""

import logging

from . import contracts
from .settings import WorkerSettings

logger = logging.getLogger("harvest.worker")

# type key -> queue name. `source` selects both the queue and (later) the handler.
QUEUE_FOR = {"arxiv": "harvest_arxiv", "openalex": "harvest_openalex", "s2": "harvest_s2"}

# Visibility timeout: how long a read message stays hidden while we work on it.
# If we don't archive within this window, pgmq redelivers it (at-least-once).
VT_SECONDS = 120


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = WorkerSettings()

    queue = QUEUE_FOR.get(settings.source)
    if queue is None:
        raise SystemExit(f"--source must be one of {sorted(QUEUE_FOR)}, got {settings.source!r}")

    supabase = settings.client()
    q = supabase.schema("pgmq_public")  # Supabase Queues wrappers over pgmq

    logger.info("draining queue %s", queue)
    drained = 0
    while True:
        resp = q.rpc("read", {"queue_name": queue, "sleep_seconds": VT_SECONDS, "n": 1}).execute()
        rows = resp.data
        if not rows:  # empty -> drain-then-exit
            break

        rec = rows[0]  # {"msg_id":…, "read_ct":…, "enqueued_at":…, "vt":…, "message": {...}}
        event = contracts.parse(rec["message"])  # {"v":1,"paper_id":N} -> SeedMessage
        logger.info(
            "read msg_id=%s paper_id=%s read_ct=%s",
            rec["msg_id"],
            event.paper_id,
            rec.get("read_ct"),
        )

        # TODO(next layer): fetch the source + upsert source_X here.

        # ack: only archive on success; leaving it unarchived redelivers after VT.
        q.rpc("archive", {"queue_name": queue, "message_id": rec["msg_id"]}).execute()
        drained += 1

    logger.info("queue %s empty — drained %d, exiting", queue, drained)
