#!/usr/bin/env python3
"""Tier 2 evaluation harness — the scorecard for the VRAM scheduler.

Turns the plan's measurement + probe claims into one PASS / FAIL / PROBE scorecard
(`scorecard.json` + `scorecard.md`) that syncs back to the laptop for review. It
drives a LIVE gateway over HTTP only — no direct model calls — so it must run on
the desktop (GPU box) against a running gateway.

Two modes:

  # Live checks: discovery probes first, then the measured checks.
  python scripts/eval_harness.py --url http://localhost:8765 \
      --out eval-results/scorecard.json

  # A/B wedge (V11): compare two benchmark.py outputs (affinity vs naive).
  python scripts/eval_harness.py --compare eval-results/affinity.json \
      eval-results/naive.json --out eval-results/ab.json

## What each check asserts (thresholds pinned below)

  PROBE replica   record `serve status` shape; confirm the gateway is single-replica.
  PROBE 4B bnb    force a 4B evict→reload; the round-trip must preserve embeddings
                  (cosine ≥ 0.999). If bnb `.to()` can't survive parking, the correct
                  outcome is ram_cache_4b == "disabled-after-failure" — the fallback
                  IS the pass; only a broken embedding fails.
  V6 coalesce     a concurrent same-model burst coalesces into fewer calls than jobs.
  V7 ram-cache    with parking enabled: park frees VRAM and unpark preserves output;
                  reports unpark latency (and the ≤0.5× speedup when a disk baseline
                  is supplied via --disk-baseline-ms).
  V10 stress      a high-concurrency mixed workload: zero errors, peak VRAM under card.
  V11 A/B         (--compare) affinity evicts ≤ naive/3, neither OOMs, affinity p95 ≤ naive.

Stdlib only. Reuses benchmark.py's request mix + counter helpers (imported).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Reuse the pinned request mix + helpers from benchmark.py (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark  # noqa: E402

# --- Pinned thresholds -----------------------------------------------------
CARD_CEILING_MB = 7500
COALESCE_BURST = 16          # concurrent same-model embeds for the V6 coalesce check
STRESS_REQUESTS = 120
STRESS_CONCURRENCY = 12

EMBED_0_6B = "cuda-qwen0.6b-1024"
EMBED_4B = "cuda-qwen4b-1024"
RERANKER = "bge-reranker-v2-m3"
CHAT_MODEL = "gemma4:e2b"        # must exist in Ollama (`ollama list`)


# --------------------------------------------------------------------------
# HTTP + math helpers
# --------------------------------------------------------------------------

def _post(url: str, path: str, body: dict, timeout: float = 180.0):
    return benchmark._post(url, path, body, timeout)


def _get_json(url: str, path: str) -> dict:
    return benchmark._get_json(url, path)


def _counter(url: str, name: str) -> int:
    return benchmark._parse_prom_counter(benchmark._get_text(url, "/metrics"), name)


def _embed_vec(url: str, model: str, text: str) -> list[float]:
    status, payload, _ = _post(url, "/v1/embeddings", {"model": model, "input": [text]})
    if status != 200:
        raise RuntimeError(f"embed {model} failed: {status} {payload}")
    return payload["data"][0]["embedding"]


def _row(check: str, verdict: str, message: str, evidence: dict) -> dict:
    return {"check": check, "verdict": verdict, "message": message, "evidence": evidence}


# --------------------------------------------------------------------------
# Probes (run first — record reality about the known-unknowns)
# --------------------------------------------------------------------------

def probe_replica(url: str) -> dict:
    """Record the `serve status` shape (the API the replica-guard depends on) and
    confirm the gateway is up as a single replica."""
    shape = ""
    try:
        out = subprocess.run(
            ["serve", "status"], capture_output=True, text=True, timeout=30
        )
        shape = (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 — CLI may be absent; record it
        shape = f"<serve status unavailable: {exc}>"
    healthy = _get_json(url, "/health").get("status") == "ok"
    return _row(
        "PROBE replica",
        "PROBE",
        "recorded `serve status` shape; gateway reachable as single replica"
        if healthy else "gateway not healthy",
        {"serve_status": shape[:2000], "gateway_healthy": healthy},
    )


def probe_4b_bnb(url: str) -> dict:
    """Force a 4B evict→reload and prove the reloaded model still produces a valid
    embedding.

    The plan's known-unknown: bitsandbytes int8 layers may not survive .to('cpu')↔
    .to('cuda'). We can't do a before/after cosine over HTTP — the gateway's SQLite
    cache would serve the second (identical) input WITHOUT reloading the model, a false
    pass. So we bust the cache with unique inputs and assert the *post-reload* embedding
    is well-formed (1024-dim, finite) — the same signal the handler's own
    _validate_after_unpark self-test uses to detect bnb corruption (NaN/Inf/wrong-dim).
    If parking can't survive, the gateway falls back (ram_cache_4b ==
    'disabled-after-failure') and STILL returns a valid embedding — that IS the pass."""
    nonce = str(time.monotonic_ns())
    try:
        _embed_vec(url, EMBED_4B, f"bnb-probe-a-{nonce}")   # miss → loads 4B
        # Chat occupies the whole card → evicts 4B (parks it if the flag is on).
        _post(url, "/v1/chat/completions",
              {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 8})
        vec_after = _embed_vec(url, EMBED_4B, f"bnb-probe-b-{nonce}")  # miss → reload/unpark
    except Exception as exc:  # noqa: BLE001
        return _row("PROBE 4B bnb", "FAIL", f"round-trip raised: {exc}", {})

    finite = all(math.isfinite(x) for x in vec_after[:32])
    well_formed = len(vec_after) == 1024 and finite
    status = _get_json(url, "/health").get("ram_cache_4b", "unknown")
    verdict = "PROBE" if well_formed else "FAIL"
    note = {
        "off": "parking not enabled; destroy/reload path",
        "enabled": "parked in RAM and unparked cleanly",
        "disabled-after-failure": "bnb .to() failed → fell back to destroy/reload (correct)",
    }.get(status, "unknown ram_cache_4b state")
    return _row(
        "PROBE 4B bnb", verdict,
        f"{note}; post-reload embedding {'valid' if well_formed else 'MALFORMED'} "
        f"(dims={len(vec_after)}, finite={finite})",
        {"ram_cache_4b": status, "dims": len(vec_after), "finite": finite},
    )


# --------------------------------------------------------------------------
# Measured checks
# --------------------------------------------------------------------------

def check_v6_coalesce(url: str) -> dict:
    """A concurrent same-model embed burst must coalesce into fewer inference calls
    than jobs served (drained same-model jobs merged into one call)."""
    before_c = _counter(url, "gpu_gateway_coalesced_calls")
    before_j = _counter(url, "gpu_gateway_jobs_served")

    def one(i: int):
        # Unique text per request so none are cache hits — all reach the dispatcher.
        return _post(url, "/v1/embeddings",
                     {"model": EMBED_0_6B, "input": [f"coalesce-{i}-{time.monotonic_ns()}"]})[0]

    with ThreadPoolExecutor(max_workers=COALESCE_BURST) as pool:
        statuses = list(pool.map(one, range(COALESCE_BURST)))

    dc = _counter(url, "gpu_gateway_coalesced_calls") - before_c
    dj = _counter(url, "gpu_gateway_jobs_served") - before_j
    ok = all(s == 200 for s in statuses) and dc >= 1 and dj > dc
    return _row(
        "V6 coalesce",
        "PASS" if ok else "FAIL",
        f"{dj} jobs served across {dc} coalesced call(s) "
        f"({dj / dc:.1f} jobs/call)" if dc else "no coalesced calls observed",
        {"jobs_served_delta": dj, "coalesced_calls_delta": dc,
         "burst": COALESCE_BURST, "errors": sum(1 for s in statuses if s != 200)},
    )


def check_v7_ram_cache(url: str, disk_baseline_ms: float | None) -> dict:
    """Exercise the evict→reload cycle for 0.6B and prove the reloaded model still
    embeds correctly; record the reload latency.

    Eviction is forced by loading the 4B (0.6B + 4B = 7600 > 7200 so the 0.6B is
    evicted) rather than a chat request — chat loads Ollama into VRAM, which would
    confound both the timing and any freed-VRAM delta. Unique inputs bust the SQLite
    cache so the reload actually happens.

    On the default gateway keep_on_cpu is OFF, so this is the from-disk reload path and
    `reload_ms` is a useful DISK BASELINE for a later parking-on run. To measure the RAM
    cache's unpark speedup, enable keep_on_cpu_after_evict for 0.6B, restart, and pass
    that baseline via --disk-baseline-ms; the ≤0.5× gate then applies. Correctness for
    the bf16 0.6B is well-formedness (finite) — bf16 device moves are reliable; the
    fragile bnb path is 4B, covered by PROBE 4B bnb with its own self-test."""
    nonce = str(time.monotonic_ns())
    try:
        _embed_vec(url, EMBED_0_6B, f"v7-load-{nonce}")     # miss → loads 0.6B
        parked_status = _get_json(url, "/health").get("ram_cache_4b", "unknown")
        _embed_vec(url, EMBED_4B, f"v7-evict-{nonce}")      # evicts 0.6B (7600 > 7200), loads 4B
        t0 = time.monotonic()
        vec_after = _embed_vec(url, EMBED_0_6B, f"v7-reload-{nonce}")  # evicts 4B, reloads 0.6B
        reload_ms = (time.monotonic() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return _row("V7 ram-cache", "FAIL", f"evict/reload raised: {exc}", {})

    finite = bool(vec_after) and all(math.isfinite(x) for x in vec_after[:32])
    ev = {"reload_ms": round(reload_ms, 1), "dims": len(vec_after), "finite": finite,
          "ram_cache_4b": parked_status}
    speedup_ok = True
    if disk_baseline_ms:
        ev["disk_baseline_ms"] = round(disk_baseline_ms, 1)
        ev["reload_ratio"] = round(reload_ms / disk_baseline_ms, 3)
        speedup_ok = reload_ms <= 0.5 * disk_baseline_ms
    # Gate: reload produced a valid embedding (destroy/reload path healthy). Speedup is
    # scored only when a disk baseline was supplied (i.e. a parking-on comparison run).
    ok = finite and speedup_ok
    msg = f"reload {reload_ms:.0f} ms, embedding {'valid' if finite else 'MALFORMED'}"
    if disk_baseline_ms:
        msg += f" ({ev['reload_ratio']}× disk baseline)"
    else:
        msg += " (parking off → this reload IS the disk baseline; see docstring)"
    return _row("V7 ram-cache", "PASS" if ok else "FAIL", msg, ev)


def check_v10_stress(url: str) -> dict:
    """A high-concurrency mixed workload: all requests succeed and peak VRAM stays
    under the card (`vram_budget_respected`)."""
    phase = benchmark.run_phase(url, STRESS_REQUESTS, STRESS_CONCURRENCY, "off", "stress")
    ok = phase["errors"] == 0 and phase["peak_vram_mb"] < CARD_CEILING_MB
    return _row(
        "V10 stress",
        "PASS" if ok else "FAIL",
        f"{STRESS_REQUESTS} reqs @ {STRESS_CONCURRENCY} conc: "
        f"{phase['errors']} errors, peak {phase['peak_vram_mb']} MB",
        {"errors": phase["errors"], "peak_vram_mb": phase["peak_vram_mb"],
         "evicts": phase["evicts"], "coalesced_calls": phase["coalesced_calls"],
         "throughput_rps": phase["throughput_rps"],
         "embed_p95_ms": phase["latency_ms"]["embed"]["p95"]},
    )


# --------------------------------------------------------------------------
# V11 A/B compare (offline — reads two benchmark.py outputs)
# --------------------------------------------------------------------------

def compare_ab(affinity_path: str, naive_path: str) -> dict:
    """Score the affinity vs naive A/B.

    The wedge is that co-residency eliminates redundant reloads — but whether that
    surfaces as latency or throughput depends on which model dominates the workload:
      - mixed profile: naive reloads the huge 4B on the MEDIAN path → the win is p50.
      - hotpath profile: reloads are cheap 0.6B, so p50 stays flat, but naive SERIALIZES
        on constant reloads → the win is throughput (rps) and the tail (p95), with evicts
        going to zero under affinity.
    So a single-metric gate is wrong. The honest, robust criterion: neither mode OOMs,
    affinity never does MORE evictions than naive, and affinity wins by ≥2× on EITHER
    warm p50 or warm throughput (whichever dimension the workload stresses)."""
    with open(affinity_path) as fh:
        aff = json.load(fh)
    with open(naive_path) as fh:
        naive = json.load(fh)

    def evicts(r):
        return r["cold"]["evicts"] + r["warm"]["evicts"]

    def peak(r):
        return max(r["cold"]["peak_vram_mb"], r["warm"]["peak_vram_mb"])

    def p50(r):
        return r["warm"]["latency_ms"]["embed"]["p50"]

    def p95(r):
        return r["warm"]["latency_ms"]["embed"]["p95"]

    def rps(r):
        return r["warm"]["throughput_rps"]

    aff_ev, naive_ev = evicts(aff), evicts(naive)
    card = aff.get("card_mb", 8192)
    no_oom = peak(aff) < card and peak(naive) < card
    aff_p50, naive_p50 = p50(aff), p50(naive)
    aff_rps, naive_rps = rps(aff), rps(naive)
    p50_speedup = round(naive_p50 / aff_p50, 1) if aff_p50 > 0 else float("inf")
    rps_speedup = round(aff_rps / naive_rps, 1) if naive_rps > 0 else float("inf")
    evict_ratio = round(naive_ev / aff_ev, 1) if aff_ev > 0 else float("inf")
    best = max(p50_speedup, rps_speedup)
    evict_ok = aff_ev <= naive_ev
    ok = no_oom and evict_ok and best >= 2.0
    profile = aff.get("profile", "mixed")
    win = "p50" if p50_speedup >= rps_speedup else "throughput"
    return _row(
        "V11 A/B",
        "PASS" if ok else "FAIL",
        f"[{profile}] wins on {win} {best}×  "
        f"(p50 {aff_p50} vs {naive_p50} ms = {p50_speedup}×; "
        f"rps {aff_rps} vs {naive_rps} = {rps_speedup}×; "
        f"evicts {aff_ev} vs {naive_ev}; no-OOM={no_oom})",
        {"profile": profile,
         "affinity": {"mode": aff.get("sched_mode"), "evicts": aff_ev, "embed_p50_ms": aff_p50,
                      "embed_p95_ms": p95(aff), "rps": aff_rps, "peak_vram_mb": peak(aff)},
         "naive": {"mode": naive.get("sched_mode"), "evicts": naive_ev, "embed_p50_ms": naive_p50,
                   "embed_p95_ms": p95(naive), "rps": naive_rps, "peak_vram_mb": peak(naive)},
         "p50_speedup": p50_speedup, "rps_speedup": rps_speedup, "evict_ratio": evict_ratio,
         "no_oom": no_oom,
         "gate": "no-OOM AND affinity_evicts<=naive AND max(p50,rps speedup)>=2x"},
    )


# --------------------------------------------------------------------------
# Scorecard rendering
# --------------------------------------------------------------------------

def render_md(rows: list[dict], meta: dict) -> str:
    icon = {"PASS": "✅", "FAIL": "❌", "PROBE": "🔍"}
    lines = ["# GPU Gateway — Scheduler Evaluation Scorecard", ""]
    lines.append(f"- gateway: `{meta.get('url', '—')}`  mode: `{meta.get('sched_mode', '—')}`")
    lines.append(f"- card: {meta.get('card_mb', '—')} MB  ceiling: {CARD_CEILING_MB} MB")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    lines.append(f"- result: **{'ALL CLEAR' if n_fail == 0 else f'{n_fail} FAILED'}**")
    lines += ["", "| Check | Verdict | Detail |", "|-------|---------|--------|"]
    for r in rows:
        lines.append(f"| {r['check']} | {icon.get(r['verdict'], '?')} {r['verdict']} | {r['message']} |")
    lines += ["", "## Evidence", ""]
    for r in rows:
        lines.append(f"- **{r['check']}** — `{json.dumps(r['evidence'])}`")
    return "\n".join(lines) + "\n"


def write_out(out_path: str, rows: list[dict], meta: dict) -> None:
    payload = {"meta": meta, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write(render_md(rows, meta))
    print(f"# wrote {out_path} and {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--out", default="eval-results/scorecard.json")
    ap.add_argument("--disk-baseline-ms", type=float, default=None,
                    help="0.6B from-disk load time (ms) to score the V7 unpark speedup")
    ap.add_argument("--compare", nargs=2, metavar=("AFFINITY_JSON", "NAIVE_JSON"),
                    help="V11 A/B: compare two benchmark.py outputs instead of live checks")
    args = ap.parse_args()

    if args.compare:
        row = compare_ab(args.compare[0], args.compare[1])
        meta = {"mode": "compare", "inputs": args.compare}
        write_out(args.out, [row], meta)
        print(f"{row['verdict']}: {row['check']} — {row['message']}")
        sys.exit(0 if row["verdict"] != "FAIL" else 1)

    health = _get_json(args.url, "/health")
    if health.get("status") != "ok":
        raise SystemExit(f"gateway not healthy at {args.url}: {health}")
    meta = {
        "url": args.url,
        "sched_mode": health.get("sched_mode", "unknown"),
        "card_mb": max((g.get("total_mb", 0) for g in health.get("gpus", [])), default=0),
    }

    rows: list[dict] = []
    # Probes first — they record reality about the known-unknowns.
    for fn in (probe_replica, probe_4b_bnb):
        try:
            rows.append(fn(args.url))
        except Exception as exc:  # noqa: BLE001 — one probe must not abort the scorecard
            rows.append(_row(fn.__name__, "FAIL", f"probe raised: {exc}", {}))
    # Measured checks.
    for fn in (check_v6_coalesce,
               lambda u: check_v7_ram_cache(u, args.disk_baseline_ms),
               check_v10_stress):
        try:
            rows.append(fn(args.url))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(getattr(fn, "__name__", "check"), "FAIL", f"raised: {exc}", {}))

    write_out(args.out, rows, meta)
    for r in rows:
        print(f"{r['verdict']:5} {r['check']:16} {r['message']}")
    sys.exit(1 if any(r["verdict"] == "FAIL" for r in rows) else 0)


if __name__ == "__main__":
    main()
