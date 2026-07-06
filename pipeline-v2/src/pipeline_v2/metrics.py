"""Prometheus metrics for the pipeline-v2 stage consumers.

Consumers run as long-lived HOST processes (one per stage/corpus, see
``pipeline_v2.cli``'s ``consume`` command), not pods — there is no sidecar or
service-mesh injection to lean on for metrics, so each consumer process
optionally exposes its own ``/metrics`` HTTP endpoint via
:func:`start_metrics_server`, scraped by the in-cluster Prometheus through a
static target (see ``deploy/observability/pipeline-v2-scrape.yaml``).

All series are labelled by ``stage`` (extract/embed/indexer) and ``corpus``
(the corpus namespace a given consumer process was started for) so Grafana
can slice/alert per stage-corpus pair without needing per-process job names.

Kept intentionally small — four counters plus one histogram, no custom
collector classes, no framework sprawl. Import this module anywhere metrics
need recording; it never binds a port on import (:func:`start_metrics_server`
is the only thing that touches the network, and only when explicitly called).
"""

from __future__ import annotations

import logging
import threading

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

_LABELS = ("stage", "corpus")

MESSAGES_PROCESSED_TOTAL = Counter(
    "pipeline_messages_processed_total",
    "Stage-consumer records for which the handler was invoked (excludes "
    "poison-message and idempotent-duplicate skips, which never reach the "
    "handler).",
    _LABELS,
)

MESSAGES_ACKED_TOTAL = Counter(
    "pipeline_messages_acked_total",
    "Records ACCEPTed off the queue (ok, degraded, or an exhausted-retry "
    "failed record that was also deadlettered).",
    _LABELS,
)

MESSAGES_RELEASED_TOTAL = Counter(
    "pipeline_messages_released_total",
    "Records RELEASEd for redelivery after a failed outcome below the "
    "dead-letter delivery-count threshold.",
    _LABELS,
)

MESSAGES_DLQ_TOTAL = Counter(
    "pipeline_messages_dlq_total",
    "Records sent to a deadletter.<stage> topic (poison messages or "
    "delivery-count-exhausted failures).",
    _LABELS,
)

HANDLER_SECONDS = Histogram(
    "pipeline_handler_seconds",
    "Wall-clock time spent inside a stage handler call (excludes contract "
    "validation, idempotency check, work_state upsert, and queue ack/produce "
    "overhead).",
    _LABELS,
)

_server_lock = threading.Lock()
_server_started_port: int | None = None


def start_metrics_server(port: int) -> None:
    """Start the ``prometheus_client`` HTTP exposition server, once.

    Idempotent: a second call with the same port is a no-op; a second call
    with a *different* port logs a warning and leaves the original server
    running (``prometheus_client`` has no supported "rebind" path, and
    silently starting a second listener would just confuse scrapers about
    which port is live). Safe to call from multiple threads/consumers within
    one process.
    """
    global _server_started_port
    with _server_lock:
        if _server_started_port is not None:
            if _server_started_port != port:
                logger.warning(
                    "metrics server already started on port %s; ignoring request for port %s",
                    _server_started_port,
                    port,
                )
            return
        start_http_server(port)
        _server_started_port = port
        logger.info("prometheus metrics server listening on :%s/metrics", port)
