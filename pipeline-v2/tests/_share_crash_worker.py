#!/usr/bin/env python
"""Subprocess helper for the crash-redelivery integration test.

Acquires exactly one record from a share group via our own
``pipeline_v2.messaging.ShareGroupQueue`` and then ``os._exit(1)`` WITHOUT
acking it — simulating a worker process dying mid-job while holding the
record's lock. Adapted from the spike's ``share_worker.py`` "crash" mode
(scratchpad/sharegroup-spike/share_worker.py), reusing our production
adapter instead of the raw client so the test also exercises
ShareGroupQueue itself.

Run under a bounded ``subprocess.run(..., timeout=...)`` from the test —
never invoked directly / unbounded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_v2.messaging import ShareGroupQueue  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    queue = ShareGroupQueue(
        bootstrap_servers=args.bootstrap,
        group_id=args.group,
        topics=[args.topic],
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        batch = queue.poll_batch(timeout=2.0)
        if batch:
            msg = batch[0]
            print(
                json.dumps(
                    {
                        "offset": msg.offset,
                        "partition": msg.partition,
                        "delivery_count": msg.delivery_count,
                        "value": msg.value.decode("utf-8"),
                    }
                ),
                flush=True,
            )
            os._exit(1)  # crash WITHOUT acking — do not let atexit/finally run
    print(json.dumps({"error": "no message acquired within timeout"}), flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
