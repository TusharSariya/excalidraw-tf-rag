#!/usr/bin/env python3
"""Reproducible mixed-traffic benchmark for the GPU gateway VRAM scheduler.

The wedge claim is that bin-packing co-residency + LRU eviction on one consumer GPU
serves mixed embed/rerank/LLM traffic with (a) no OOM, (b) far fewer model reloads
than a naive load-evict baseline, and (c) competitive latency. This script produces
the artifact that proves (or refutes) it.

It drives a *pinned* workload against a running gateway and reports:
  - peak VRAM used (sampled from /health; must stay under the card)
  - eviction / admit counters (from /metrics) — the reload proxy
  - p50 / p95 / p99 request latency, split by request kind

Everything is pinned for reproducibility (Codex #19): request mix, arrival concurrency,
payload sizes, a cold phase then a warm phase, and cache on/off. Deterministic inputs
(seeded, no wall-clock randomness) so two runs are comparable.

## A/B vs the naive baseline

This measures ONE gateway. To get the naive-load-evict baseline for comparison, run
the same script against a gateway started from the pre-scheduler commit:

    git stash                             # or check out the parent of the scheduler change
    # (re)start the gateway on the desktop
    python scripts/benchmark.py --url http://localhost:8765 --tag naive --out naive.json
    git stash pop                         # restore the scheduler
    # restart the gateway
    python scripts/benchmark.py --url http://localhost:8765 --tag scheduler --out scheduler.json

Then compare evicts and p95 between naive.json and scheduler.json. The headline number
is evicts: the scheduler should show ~1 eviction per model-batch instead of per-request.

Usage:
    python scripts/benchmark.py --url http://localhost:8765 [--requests 200] [--concurrency 8]
                                [--cache off|on] [--out results.json] [--tag NAME]

Stdlib only — runs anywhere with python3, no gateway deps needed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unique per process (= per benchmark run) so cache='off' inputs never collide with a
# previous run's SQLite cache. Overridable via --run-salt for exact reproduction.
RUN_SALT = os.urandom(3).hex()

# Pinned request mixes (weights = relative frequency). Ordered so the sequence is
# deterministic across runs. Two profiles:
#   mixed   — realistic: 4B + LLM force full-card eviction in BOTH modes, so the wedge
#             here is median latency (warm co-residency), not reload count.
#   hotpath — the RAG embed→rerank path in isolation (no 4B/LLM). Co-residency dominates:
#             0.6B+reranker (3000 MB) stay resident under affinity but thrash under naive,
#             so THIS is where the reload-count win shows up.
MIX_PROFILES = {
    "mixed": [
        ("embed", "cuda-qwen0.6b-1024", 5),   # the hot RAG path
        ("embed", "cuda-qwen4b-1024", 2),     # heavy embedder — forces eviction pressure
        ("rerank", "bge-reranker-v2-m3", 2),  # follows embed → tests co-residency
        ("chat", "gemma4:e2b", 1),            # LLM → evicts everything (must exist in Ollama)
    ],
    "hotpath": [
        ("embed", "cuda-qwen0.6b-1024", 3),   # embed→rerank alternation, co-resident pair
        ("rerank", "bge-reranker-v2-m3", 2),
    ],
}
MIX = MIX_PROFILES["mixed"]        # default; overridden per --profile in main()

EMBED_TEXT_CHARS = 400   # ~100 tokens; pinned payload size
RERANK_PASSAGES = 8
CHAT_PROMPT = "Summarize graph layout constraints in one sentence."


def _post(url: str, path: str, body: dict, timeout: float = 180.0) -> tuple[int, dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
            return resp.status, payload, time.monotonic() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")}, time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}, time.monotonic() - t0


def _get_json(url: str, path: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return {}


def _get_text(url: str, path: str) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=10) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def _parse_prom_counter(text: str, name: str) -> int:
    for line in text.splitlines():
        if line.startswith(name + " "):
            try:
                return int(float(line.split()[1]))
            except (ValueError, IndexError):
                return 0
    return 0


def build_request(i: int, kind: str, model: str, cache: str) -> tuple[str, dict]:
    """Deterministic request builder. cache='off' → unique text per request (all miss);
    cache='on' → reuse a small pool so most hit.

    RUN_SALT makes cache='off' inputs unique PER RUN — without it, a second run (e.g. the
    naive A/B phase) would hit the first run's SQLite cache and never touch the dispatcher.
    The workload SHAPE (sizes, sequence) stays identical across runs; only the bytes differ."""
    salt = "on" if cache == "on" else f"{RUN_SALT}:{i}"
    pool = i % 4 if cache == "on" else i
    if kind == "embed":
        text = f"[{pool}:{salt}] " + ("graph layout constraint solver " * 12)[:EMBED_TEXT_CHARS]
        return "/v1/embeddings", {"model": model, "input": [text]}
    if kind == "rerank":
        q = f"query {pool} {salt}"
        passages = [f"passage {pool} {salt} {j} " + ("x" * 30) for j in range(RERANK_PASSAGES)]
        return "/v1/rerank", {"model": model, "query": q, "passages": passages}
    # chat
    return "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": CHAT_PROMPT}],
        "max_tokens": 32,
    }


def make_sequence(n: int) -> list[tuple[str, str]]:
    """Expand MIX weights into a deterministic round-robin sequence of length n."""
    expanded: list[tuple[str, str]] = []
    for kind, model, w in MIX:
        expanded.extend([(kind, model)] * w)
    seq = []
    for i in range(n):
        seq.append(expanded[i % len(expanded)])
    return seq


class PeakSampler(threading.Thread):
    """Polls /health for peak used_mb until stopped."""

    def __init__(self, url: str, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.peak_mb = 0
        self.samples = 0
        self._halt = threading.Event()   # NOT _stop: that shadows Thread._stop(), which join() calls

    def run(self) -> None:
        while not self._halt.is_set():
            h = _get_json(self.url, "/health")
            for g in h.get("gpus", []):
                self.peak_mb = max(self.peak_mb, int(g.get("used_mb", 0)))
            self.samples += 1
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


def run_phase(url: str, n: int, concurrency: int, cache: str, phase: str) -> dict:
    seq = make_sequence(n)
    lat: dict[str, list[float]] = {"embed": [], "rerank": [], "chat": []}
    errors = 0

    sampler = PeakSampler(url)
    sampler.start()
    before = _get_text(url, "/metrics")

    def one(i: int) -> tuple[str, int, float]:
        kind, model = seq[i]
        path, body = build_request(i, kind, model, cache)
        status, _, dt = _post(url, path, body)
        return kind, status, dt

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, i) for i in range(n)]
        for f in as_completed(futs):
            kind, status, dt = f.result()
            if status == 200:
                lat[kind].append(dt)
            else:
                errors += 1
    wall = time.monotonic() - t0

    sampler.stop()
    sampler.join(timeout=2)
    after = _get_text(url, "/metrics")

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return round(xs[k] * 1000, 1)  # ms

    return {
        "phase": phase,
        "requests": n,
        "concurrency": concurrency,
        "cache": cache,
        "errors": errors,
        "wall_s": round(wall, 2),
        "throughput_rps": round(n / wall, 1) if wall else 0,
        "peak_vram_mb": sampler.peak_mb,
        "latency_ms": {
            k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)}
            for k, v in lat.items()
        },
        "evicts": _parse_prom_counter(after, "gpu_gateway_evicts")
        - _parse_prom_counter(before, "gpu_gateway_evicts"),
        "admits": _parse_prom_counter(after, "gpu_gateway_admits")
        - _parse_prom_counter(before, "gpu_gateway_admits"),
        "coalesced_calls": _parse_prom_counter(after, "gpu_gateway_coalesced_calls")
        - _parse_prom_counter(before, "gpu_gateway_coalesced_calls"),
        "jobs_served": _parse_prom_counter(after, "gpu_gateway_jobs_served")
        - _parse_prom_counter(before, "gpu_gateway_jobs_served"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cache", choices=["on", "off"], default="off",
                    help="off = every request misses (stresses the scheduler); on = warm cache")
    ap.add_argument("--tag", default="run", help="label for this run (e.g. naive | scheduler)")
    ap.add_argument("--profile", choices=list(MIX_PROFILES), default="mixed",
                    help="mixed = realistic 4B+LLM load; hotpath = embed→rerank co-residency only")
    ap.add_argument("--run-salt", default="", help="override the per-run cache-buster (reproducibility)")
    ap.add_argument("--out", default="", help="write JSON results to this path")
    args = ap.parse_args()
    if args.run_salt:
        global RUN_SALT
        RUN_SALT = args.run_salt
    global MIX
    MIX = MIX_PROFILES[args.profile]

    health = _get_json(args.url, "/health")
    if health.get("status") != "ok":
        raise SystemExit(f"gateway not healthy at {args.url}: {health}")
    card_mb = max((g.get("total_mb", 0) for g in health.get("gpus", [])), default=0)
    # The scheduler mode the gateway is running (affinity | naive) labels the A/B curve.
    sched_mode = health.get("sched_mode", "unknown")

    print(f"# benchmark tag={args.tag} profile={args.profile} mode={sched_mode} "
          f"url={args.url} card={card_mb} MB")
    # Cold phase: models unloaded → measures worst-case load/evict behavior.
    cold = run_phase(args.url, max(20, args.requests // 4), args.concurrency, args.cache, "cold")
    # Warm phase: models resident → measures steady-state throughput + thrash.
    warm = run_phase(args.url, args.requests, args.concurrency, args.cache, "warm")

    result = {"tag": args.tag, "sched_mode": sched_mode, "profile": args.profile,
              "url": args.url, "card_mb": card_mb, "cold": cold, "warm": warm}

    def line(p: dict) -> str:
        emb = p["latency_ms"]["embed"]
        return (f"  {p['phase']:5} reqs={p['requests']:4} err={p['errors']:3} "
                f"peakVRAM={p['peak_vram_mb']:5} evicts={p['evicts']:3} admits={p['admits']:3} "
                f"coalesced={p['coalesced_calls']:3} rps={p['throughput_rps']:5} "
                f"embed_p50/p95={emb['p50']}/{emb['p95']}ms")

    print(line(cold))
    print(line(warm))
    over = [p for p in (cold, warm) if card_mb and p["peak_vram_mb"] >= card_mb]
    print(f"# peak-VRAM-safe: {'NO — OOM RISK' if over else 'yes'} "
          f"(peak {max(cold['peak_vram_mb'], warm['peak_vram_mb'])} / {card_mb} MB)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
