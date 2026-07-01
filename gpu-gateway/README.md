# GPU Gateway

A persistent inference gateway that serves **embedding, reranking, and LLM models
through one VRAM budget** on a single consumer GPU — with LRU eviction *and*
bin-packing co-residency so models that fit stay loaded together.

## The wedge

Serving retrieval-augmented generation on an 8 GB card (RTX 3060 Ti) means juggling
three *different kinds* of model — dense embedders (0.6B, 4B), a cross-encoder
reranker, and an LLM — that together far exceed VRAM. Ollama is LLM-only; Triton/vLLM
assume datacenter cards; LocalAI lazy-loads with no principled eviction. None of them
treat embed + rerank + LLM as **one budget** on a consumer GPU.

This gateway does: point it at an 8 GB card, throw mixed embed/rerank/LLM traffic at it,
and it (a) never OOMs, (b) keeps co-resident models warm — `0.6B + reranker = 3000 MB`
both stay loaded, so the common embed→rerank path never thrashes — and (c) evicts only
under real pressure, LRU with a queue-depth tie-break. All of it behind an
OpenAI-compatible HTTP surface.

## Architecture

`GatewayHTTP` is a single Ray Serve replica. All GPU work funnels through one
**in-process dispatcher** (`scheduler.py`) that is the only code touching VRAM:

- **Atomic eviction.** Eviction and the load that follows are one sequential step in a
  single background task — no TOCTOU window where two requests both pass the budget
  check and breach the ceiling.
- **Bin-packing co-residency.** A model that fits alongside what's already loaded is
  simply added; eviction happens only when the budget (and a **measured-VRAM gate**)
  says it must. The measured gate catches cases the static weights table misses (CUDA
  context + activation memory).
- **Throughput scheduling.** Jobs queue per model; the dispatcher serves
  resident-with-work first, then the deepest queue, coalescing a same-model batch into
  one inference call. A `STARVE_MS` fairness floor bounds worst-case wait.
- **Opt-in RAM cache.** Evicted models can be parked in CPU RAM (`.to('cpu')`) for a
  fast reload instead of destroyed (per-model flag, default off; the 4B int8 path
  self-tests after unpark and falls back to destroy/reload if bitsandbytes misbehaves).

Handlers (`handlers/`) are separate Ray Serve deployments that lazily load their model
on first call and expose an `unload()` that parks or destroys; the eviction mechanics
are shared via the `_EvictableModel` mixin.

## Quickstart

```bash
cd tools/gpu-gateway
uv sync
cp .env.example .env      # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch with CUDA is installed separately, see pyproject.toml
serve run config/serve_config.yaml
```

Then, from any client:

```bash
# embed
curl -s localhost:8765/v1/embeddings \
  -d '{"model":"cuda-qwen0.6b-1024","input":["hello world"]}'
# rerank
curl -s localhost:8765/v1/rerank \
  -d '{"model":"bge-reranker-v2-m3","query":"q","passages":["a","b"]}'
# chat (proxied to Ollama; evicts embedders first)
curl -s localhost:8765/v1/chat/completions \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"hi"}]}'
```

Observability:

```bash
curl -s localhost:8765/health    # resident/parked models, queue depth, counters, ram_cache_4b
curl -s localhost:8765/metrics   # Prometheus-style: evicts, admits, coalesced_calls, queue_depth
```

## Evaluation

Three tiers, one per run environment. Every claim in the design is an executable
**PASS / FAIL / PROBE** check.

### Tier 0 — unit (no GPU, runs anywhere)

The scheduler's decision logic is pure (no ray/torch), so the unit suite runs with no
GPU. Integration tests auto-skip without a gateway, so a plain run stays laptop-safe:

```bash
PYTHONPATH=src uv run --no-project --with pytest --with pytest-asyncio pytest tests/
```

### Tier 1 — gated correctness (live gateway, desktop)

`tests/test_integration.py` turns the correctness claims into fail-closed assertions
against a running gateway (V1 no VRAM breach, V2 co-residency without thrash, V4
no starvation, V5 sequential-load stays under budget, V9 LLM residency clears):

```bash
PYTHONPATH=src uv run --no-project --with pytest --with pytest-asyncio \
  pytest -m integration --gateway-url http://localhost:8765 -v
```

### Tier 2 — scorecard + wedge proof (live gateway, desktop)

`scripts/eval_harness.py` runs the discovery **probes first** (`serve status` shape;
the 4B bitsandbytes `.to()` round-trip — where the *fallback* is the pass), then the
measured checks (V6 coalescing, V7 RAM cache, V10 stress), and writes
`eval-results/scorecard.json` + `scorecard.md`:

```bash
python scripts/eval_harness.py --url http://localhost:8765 --out eval-results/scorecard.json
```

**The A/B wedge proof (V11)** compares the real scheduler against a naive load-evict
baseline produced by the *same binary* via `GATEWAY_SCHED_MODE=naive` (single-resident,
drain=1, no co-residency). It's a two-phase run — Ray Serve can't flip co-residency
mid-flight — so restart the gateway between phases. Run **two profiles**, because the
win changes shape with the workload:

- `--profile mixed` (default): realistic load with 4B + LLM. Those each need (nearly) the
  whole card, so they force eviction in *both* modes — the win here is **median latency**
  (naive reloads the huge 4B on the median path).
- `--profile hotpath`: the embed→rerank path in isolation. Co-residency dominates, so the
  win is **throughput** and **evicts** (0.6B+reranker stay resident vs constant thrash).

```bash
# Phase A — the real scheduler (run BOTH profiles)
python scripts/benchmark.py --url ... --cache off --profile mixed   --tag affinity     --out eval-results/affinity.json
python scripts/benchmark.py --url ... --cache off --profile hotpath --tag affinity_hot --out eval-results/affinity_hot.json
# ...restart the gateway with GATEWAY_SCHED_MODE=naive, then re-run both as naive/naive_hot...
# Score each delta. Gate: no OOM either side, affinity never evicts more than naive,
# and affinity wins ≥2× on EITHER warm p50 or throughput (whichever the workload stresses).
python scripts/eval_harness.py --compare eval-results/affinity.json     eval-results/naive.json     --out eval-results/ab_mixed.json
python scripts/eval_harness.py --compare eval-results/affinity_hot.json eval-results/naive_hot.json --out eval-results/ab_hotpath.json
```

Measured on the RTX 3060 Ti: **mixed → 155× better median embed latency; hotpath → 23×
throughput and 0 vs 9 evictions** — no OOM in any run. `benchmark.py` reads `sched_mode`
from `/health` (each curve self-labels) and salts inputs per run so a re-run can't be
served from the SQLite cache. `eval-results/` is gitignored; sync `scorecard.md` back to
the laptop to review.

## VRAM budget

Measured on the RTX 3060 Ti (8 GB), in `main.py`:

| model | weights (MB) |
|-------|-------------|
| `cuda-qwen4b-1024` (int8, 1024-dim MRL) | 5800 |
| `cuda-qwen0.6b-1024` (bf16) | 1800 |
| `bge-reranker-v2-m3` | 1200 |

Card ceiling `_CARD_MB = 7200` (leaves ~600 MB for driver/CUDA). Co-resident pairs that
fit: `0.6B + reranker` (3000), `4B + reranker` (7000, tight — the measured gate guards
it). `0.6B + 4B` (7600) does **not** fit and is refused. `GatewayHTTP` must stay at
`num_replicas: 1` (it fails closed otherwise).
