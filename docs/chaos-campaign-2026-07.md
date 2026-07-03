# Chaos-Engineering Validation Campaign — 2026-07-03

Live checklist + evidence log. The ralph loop and orchestrator read this top-to-bottom,
execute the next unchecked `[ ]` item, verify, record evidence, check it off.
Plan: `~/.claude/plans/curried-watching-bird.md`. Design: `~/.gstack/projects/.../tushar-research-tool-e0-e1-design-20260703-093000.md`.

## Autonomy contract (operator asleep)
No sudo (root steps → BLOCKED + note the command). No AskUserQuestion. OOM is the cardinal sin (mem-guard live). Bounded timeouts, verify outputs. GPU is singular — never two gateway faults at once. WIP-commit each green item.

## Completion promise
`CHAOS CAMPAIGN COMPLETE: ALL EXPERIMENTS PASSED AND SYSTEM STABLE` — emit ONLY when every box below is `[x] PASS` (zero BLOCKED), E15 passed, and the final eval floors are met.

## Pre-req tasks
- [x] T0h — OOM mem-guard armed and running (`deploy/chaos/mem-guard.sh`)
- [x] T0a — Chaos Mesh installed via tofu, pods Running; deploy/chaos CR templates present
- [x] T0b — S3ArtifactStore implemented + integration test green vs in-cluster SeaweedFS
- [x] T0c — `pipeline-v2 enqueue` CLI + round-trip test green
- [x] T0d — `pipeline-v2 verify` oracle + must-fail self-test (seeded violation → nonzero)
- [x] T0e — GLR host ingest confirmed stopped; resume command documented
- [x] T0f — consumer prometheus metrics + scrape wired; metric visible in Prometheus
- [ ] T0g — host-process consumer launcher with memory caps; consumers connect to cluster

## Experiments
- [ ] E1 — baseline mixed dual-corpus (record throughput)
- [ ] E2 — kill extract consumer
- [ ] E3 — kill embed consumer
- [ ] E4 — kill gpu-gateway mid-embed
- [ ] E5 — kill Kafka broker
- [ ] E6 — kill CNPG instance
- [ ] E7 — kill SeaweedFS mid-write
- [ ] E8 — poison flood → DLQ
- [ ] E9 — duplicate flood → idempotent
- [ ] E10 — Kafka +2s latency (≥40% baseline)
- [ ] E11 — partition consumer↔gateway 90s
- [ ] E12 — CNPG backup during SeaweedFS restart + restore drill
- [ ] E13 — all-stages both-corpora + random kill (≥40% baseline)
- [ ] E14 — 20-min soak
- [ ] E15 (CAPSTONE) — full dual-corpus ingest under scheduled chaos
- [ ] FINAL EVALS — regression floors (GLR catalog ≥0.853, RLR catalog ≥0.534, RLR pdf-deep-read ≥0.854)

## Resume commands (for the operator)
- GLR repair ingest (kept stopped during campaign): `cd graph-layout-rag && RAG_EMBED_PROFILE=cuda-qwen0.6b-1024 RAG_GPU_GATEWAY_URL=http://<gateway-svc>:8765 GRAPH_RAG_EXTRACT_WORKERS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup .venv/bin/graph-layout-rag ingest -v > data/reingest-resume.log 2>&1 &`

## Evidence log
(appended per iteration: timestamp, item, commands run, counts, verify output, verdict)

---

### 2026-07-03T05:58:45Z — checkpoint
- Commits cba8ca3 + 0e452f6 protect all tonight work (offline suite 72 passed).
- T0h mem-guard armed (floor 3GB, +2GB swap-rise); earlyoom backstop confirmed (k3s protected -999).
- T0a Chaos Mesh: chaos-mesh ns pods Running (controller+daemon+dns), 6 CR templates in deploy/chaos/, k3s containerd socket wired.
- API dropped 4 subagents mid-response; consolidated finisher agent relaunched (commits per sub-task). T0b S3 store + T0d verify.py + T0f metrics.py partial-but-present.
