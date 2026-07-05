"""Structured logging for harvest (console + rotating file)."""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import signal
import sys
from pathlib import Path

from rag_literature_rag.paths import DATA_DIR

HARVEST_LOG_PATH = DATA_DIR / "harvest.log"
_LOGGER: logging.Logger | None = None


def _enable_hang_diagnostics() -> None:
    """`kill -USR1 <pid>` dumps all thread stacks to stderr.

    Hang forensics (2026-07-04 pass-4 wedge): when a harvest looks stuck,
    send SIGUSR1 for a full thread dump in the console/nohup output instead
    of killing blind. Main thread only; no-op where SIGUSR1 is unavailable.
    """
    with contextlib.suppress(ValueError, AttributeError, OSError):
        faulthandler.register(signal.SIGUSR1, all_threads=True)


def setup_harvest_logging(
    *,
    log_file: Path | None = None,
    verbose: bool = False,
) -> logging.Logger:
    global _LOGGER
    path = log_file or HARVEST_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag_literature_rag.harvest")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _LOGGER = logger
    _enable_hang_diagnostics()
    logger.info("logging to %s (verbose=%s)", path, verbose)
    return logger


def get_logger() -> logging.Logger:
    if _LOGGER is None:
        return setup_harvest_logging()
    return _LOGGER
