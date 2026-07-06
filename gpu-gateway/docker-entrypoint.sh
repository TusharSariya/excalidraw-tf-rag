#!/usr/bin/env bash
# gpu-gateway/docker-entrypoint.sh
#
# Mirrors systemd/gpu-gateway.service's ExecStartPre (start the Ray head if it
# isn't already up) + ExecStart (`serve run`), collapsed into one process tree
# so the container's PID 1 is `serve run` (foreground, receives SIGTERM cleanly
# from `docker stop` / kubelet).
set -euo pipefail

RAY_ARGS=(--head --disable-usage-stats --num-cpus "${RAY_NUM_CPUS:-4}")

if [[ -n "${RAY_NUM_GPUS:-}" ]]; then
  # Override the GPU count Ray schedules against. Only meant for a CPU-only
  # smoke test on a machine with no physical GPU: every handler
  # (DenseEmbed4B/06B, RerankerDeployment) loads its model LAZILY on first
  # request (see src/gpu_gateway/handlers/*.py), so letting Ray *schedule*
  # the fractional-GPU deployments doesn't touch CUDA at container boot —
  # only the first embed/rerank call would. Leave this unset in production;
  # under the NVIDIA GPU Operator / `docker run --gpus`, Ray auto-detects the
  # real GPU count from the NVIDIA container runtime.
  RAY_ARGS+=(--num-gpus "${RAY_NUM_GPUS}")
fi

ray start "${RAY_ARGS[@]}"

# `serve run` blocks in the foreground and is the container's main process.
exec serve run config/serve_config.yaml
