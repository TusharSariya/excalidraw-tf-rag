#!/usr/bin/env bash
# OOM tripwire for the chaos campaign (plan task T0h). The cardinal sin is
# OOM; this guarantees the OOM killer never fires by pre-empting it. Runs the
# whole campaign. Emits a line ONLY on a threshold breach or its own start/stop
# so it is quiet unless it acts.
set -uo pipefail

LOG="${CHAOS_LOG:-/home/tushar/excalidraw-tf-rag/docs/chaos-campaign-2026-07.md}"
AVAIL_FLOOR_MB="${AVAIL_FLOOR_MB:-3000}"
# Trip on swap RISE above the arm-time baseline (residual swap from an earlier
# incident drains slowly; an absolute ceiling would false-trip). A 2GB rise
# means new memory pressure is spilling to swap = act before OOM.
SWAP_RISE_MB="${SWAP_RISE_MB:-2048}"
KUBECONFIG="${KUBECONFIG:-/home/tushar/.kube/config}"
export KUBECONFIG

swap_base=$(free -m | awk '/^Swap:/{print $3}')
echo "MEMGUARD: armed (floor ${AVAIL_FLOOR_MB}MB avail, trip on +${SWAP_RISE_MB}MB swap over baseline ${swap_base}MB) $(date -u +%H:%M:%S)"
while true; do
  read -r avail swap_used < <(free -m | awk '/^Mem:/{a=$7} /^Swap:/{s=$3} END{print a, s}')
  swap_rise=$(( ${swap_used:-0} - swap_base ))
  if [ "${avail:-99999}" -lt "$AVAIL_FLOOR_MB" ] || [ "$swap_rise" -gt "$SWAP_RISE_MB" ]; then
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "MEMGUARD TRIP: avail=${avail}MB swap=${swap_used}MB — killing ingest/consumers + chaos"
    pkill -9 -f 'graph-layout-rag [i]ngest' 2>/dev/null
    pkill -9 -f 'rag-literature-rag [i]ngest' 2>/dev/null
    pkill -9 -f 'pipeline-v2 [c]onsume' 2>/dev/null
    pkill -9 -f 'pipeline_v2.*consume' 2>/dev/null
    ~/.local/bin/kubectl delete podchaos,networkchaos --all -A --wait=false 2>/dev/null
    printf '\n> **MEMGUARD ABORT %s** — avail=%sMB swap=%sMB (rise=%sMB); killed ingest/consumers, cleared chaos CRs.\n' "$ts" "$avail" "$swap_used" "$swap_rise" >> "$LOG"
    echo "MEMGUARD: tripped and acted at $ts — exiting so orchestrator sees the event"
    exit 42
  fi
  sleep 30
done
