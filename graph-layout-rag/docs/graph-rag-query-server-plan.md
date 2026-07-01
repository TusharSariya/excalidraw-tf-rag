# Graph-layout-RAG query server — design plan

Status: **proposed** (no code yet) · Author: design pass 2026-06-29 · Owner: Tushar

## Why

Today every query SSHes into the desktop GPU box and runs `uv run graph-layout-rag query …` as a **fresh process that cold-loads the embedding model from scratch** (observed: the NL/4B path loads all 398 weight shards before answering a single query). There is **no admission control**: N agents firing concurrently open N SSH sessions, each spawning a process that tries to load a model onto the single 8 GB RTX 3060 Ti. Under a multi-agent fan-out this OOMs / thrashes the GPU — the "weirdness" we want to eliminate.

A persistent HTTP daemon on the desktop fixes three things at once:

| Problem today | Server fix |
| --- | --- |
| Cold model load _per query_ (dominant latency) | Model resident in VRAM; load once, serve many |
| Unbounded concurrent GPU loads → OOM | Single in-flight GPU slot; others queue and wait |
| No visibility into who/what/how-slow | One choke point that logs every call |

It also lets **both** regimes (keyword 0.6B + NL 4B) serve from the box, which satisfies the original "route everything to the box" goal cleanly. Both models fit resident: 4B-4bit ≈ ~3 GB + 0.6B ≈ ~1 GB ≈ ~4 GB on an 8 GB card (idle now at 129 MiB).

## Decisions locked (2026-06-29)

- **Transport:** Tailscale / LAN. _(Tailscale is not yet installed on the desktop; LAN `10.0.0.156` is reachable today and is the same host as the `desktop` SSH alias. Phase 0 ships on LAN; Tailscale is an optional follow-up for off-LAN access.)_
- **GPU admission control:** serialize — **one** GPU embed in flight, others queue.
- **Model residency:** keep warm, **unload after idle timeout** to free VRAM.
- **Process:** write this plan first, then build.

## Architecture

```
  Mac (1..N agents)                         Desktop 10.0.0.156 (RTX 3060 Ti, 8 GB)
  ┌─────────────────┐                       ┌────────────────────────────────────┐
  │ query_auto.sh   │   HTTP POST /query    │  uvicorn (FastAPI)                  │
  │ query_remote.sh │ ────────────────────► │   ├─ asyncio.Semaphore(1)  ← GPU    │
  │  (curl client)  │ ◄──────────────────── │   ├─ warm RetrieveContext cache     │
  └─────────────────┘   JSON results        │   │    (0.6B + 4B, idle-unload)     │
                                            │   ├─ search() / retrieve_candidates │
                                            │   └─ request log → SQLite           │
                                            └────────────────────────────────────┘
```

The server is a thin async wrapper over the **existing** query path — no retrieval logic is reimplemented:

- `graph_layout_rag.query.search.search(...)` (cli.py:534 calls this) is the entry.
- `resolve_retrieve_context(embed_profile=…)` already **caches** the embed config, models, and LanceDB tables per-profile in module state. A long-lived process therefore stays warm for free; the idle-unload timer just calls the existing `clear_retrieve_caches()`.

### Endpoints

- `POST /query` — body `{ text, mode: "auto|keyword|nl", top, sparse_weight?, pdf_only?, filters?…, client_id? }`. Returns the same JSON shape the CLI emits today (`{query, results:[…]}`), so callers are drop-in. `mode:"auto"` runs the existing `route` classifier server-side and picks the profile.
- `GET /healthz` — liveness + which models are currently resident + VRAM in use + queue depth. Cheap, no GPU.
- `GET /stats?since=…` — rollup from the request log (counts, p50/p95 latency by mode and by client). Optional, Phase 2.

### Concurrency & queue

- A single `asyncio.Semaphore(1)` guards the **GPU embed** step (the only VRAM-contended part). Retrieval/fusion (BM25 + LanceDB) is CPU/RAM and may run outside the semaphore if profiling shows it matters; start simple — whole request under the semaphore.
- Requests **block in the queue** rather than 429, so a 12-agent burst drains in order instead of failing — friendlier for multi-agent callers. Add a bounded `max_queue` + `queue_timeout` so a wedged GPU returns a clean 503 instead of hanging forever.

### Warm + idle-unload

- Models load lazily on first use of each profile (keyword vs NL), then stay resident.
- A background task tracks `last_used` per profile; after `GRAPH_RAG_SERVER_IDLE_UNLOAD_S` (default e.g. 1800 s) of no use it calls `clear_retrieve_caches()` to free VRAM. Next query pays one cold load.

### Analytics (request log)

SQLite table `query_log` on the desktop, one row per request:

| col | example |
| --- | --- |
| `ts` | 2026-06-29T12:00:00Z |
| `client_id` | "agent-7" / "tushar-cli" (from body or `X-Client-Id`) |
| `mode` | auto→nl |
| `query` | "why do layered drawings…" |
| `top` | 8 |
| `n_results` | 8 |
| `queued_ms` | time spent waiting for the semaphore |
| `embed_ms` / `retrieve_ms` / `total_ms` | latency breakdown |
| `vram_mb` | nvidia-smi reading at serve time |
| `cold_load` | bool — did this request pay a model load |
| `status` | ok / timeout / error |

`queued_ms` + `cold_load` are the two columns that will tell us _when_ the box is saturated and whether the warm cache is doing its job.

## Mac client changes

- `scripts/query_auto.sh`: replace the keyword `exec uv run … query` and the NL `exec … query_remote.sh` branches with a single `curl` to `http://$GRAPH_RAG_SERVER_URL/query` passing `mode`. Keep a **fallback**: if the server is unreachable, keyword falls back to local 0.6B (preserves today's offline-keyword behavior); NL fails loud as it does now.
- New env in `.env` / `.env.example`: `GRAPH_RAG_SERVER_URL=http://10.0.0.156:8porthere` and a `GRAPH_RAG_CLIENT_ID` so multi-agent callers self-identify.
- `query_remote.sh` keeps its parity preflight conceptually, but the check moves server-side (the server validates index parity once at startup, not per query).

## Ops

- `systemd --user` (or system) unit `graph-rag-query.service` on the desktop: `uv run uvicorn graph_layout_rag.server:app --host 0.0.0.0 --port …`, `Restart=on-failure`, start on boot. Bind to `0.0.0.0` for LAN; firewall-scope to the LAN/tailnet.
- Logs to journald; request log to `data/query_log.sqlite`.
- Optional Phase: install Tailscale on the desktop and bind/advertise the tailnet IP for off-LAN agent access.

## Failure modes & guards

- **GPU OOM despite semaphore(1):** shouldn't happen with serialized access + both models ≈4 GB, but the healthz VRAM reading + `cold_load` log will catch regressions. Hard cap: refuse to load the 4B if free VRAM < threshold → 503.
- **Server down:** Mac keyword path falls back to local; NL fails loud (unchanged contract). `/healthz` from the Mac before a fan-out is cheap insurance.
- **Stale index after a rebuild:** server validates index parity (chunk count / model / dims) at startup and refuses to serve a half-synced index — same HG2 guard `query_remote.sh` does today, just lifted to startup.

## Rollout phases

- **Phase 0 — MVP:** FastAPI `POST /query` + `/healthz`, semaphore(1), warm cache, both profiles, request log, systemd unit, LAN bind. Mac `query_auto.sh` → curl with local-keyword fallback. _This alone fixes the overwhelm._
- **Phase 1 — idle-unload + parity-at-startup + queue bounds/timeout.**
- **Phase 2 — `/stats` analytics rollup; optional Tailscale; optional per-client rate limits if one agent hogs the queue.**

## Open questions for review

1. Port number + bind scope (LAN-only firewall rule vs Tailscale-only)?
2. Idle-unload timeout value — 30 min reasonable, or keep-warm-forever since the desktop is dedicated?
3. Do we want keyword to _also_ go to the box (consistency: box CUDA-embed vs Mac MLX-embed give different top results), or keep keyword local for latency and use the box only for NL? (The server supports both; this is a routing-default choice.)
4. Auth: is LAN/tailnet trust enough, or do we want a shared-secret header?
