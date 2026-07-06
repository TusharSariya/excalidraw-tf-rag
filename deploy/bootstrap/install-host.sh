#!/usr/bin/env bash
# deploy/bootstrap/install-host.sh
#
# THE ONE SCRIPT THAT NEEDS ROOT. Run it yourself with sudo:
#
#   sudo deploy/bootstrap/install-host.sh
#
# What it does, in order:
#   1. Installs k3s (single-node, pinned version) via get.k3s.io.
#   2. Installs the NVIDIA Container Toolkit (host-level) and points k3s's
#      embedded containerd at the nvidia runtime, so pods can request
#      nvidia.com/gpu once the GPU Operator's device plugin is deployed
#      (deploy/tofu/gpu-operator.tf).
#   3. Runs sanity checks: nvidia-smi, `kubectl get nodes`.
#
# Idempotent: safe to re-run. k3s's installer no-ops on a matching version;
# steps that would otherwise fail on a second run are guarded.
#
# Explicitly NOT done here (left to deploy/tofu + kustomize, layer above):
#   - GPU Operator, CloudNativePG, Strimzi, SeaweedFS, Grafana LGTM.
#   - Any application/service deployment.
#
# Verified against:
#   - https://docs.k3s.io/quick-start  (get.k3s.io installer, INSTALL_K3S_VERSION)
#   - https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
#   - https://github.com/NVIDIA/nvidia-container-toolkit install docs (apt path)
#
# NOT verified live (no root in the authoring session) — re-check output of
# each step against the printed "sanity check" section before trusting it.

set -euo pipefail

# ---------------------------------------------------------------------------
# Pinned versions. Bump deliberately; do not float on "latest" or "stable"
# channel drift for a system you have to debug at 2am.
# ---------------------------------------------------------------------------
readonly K3S_VERSION="v1.35.5+k3s1"          # k3s "stable" channel as of 2026-07-02.
readonly K3S_CHANNEL="stable"                 # informational; INSTALL_K3S_VERSION pin wins.
readonly NVIDIA_CTK_MIN_VERSION="1.17"        # nvidia-container-toolkit apt package floor.

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
warn() { printf '\n\033[1;33m!!\033[0m %s\n' "$1" >&2; }
die() { printf '\n\033[1;31mFATAL\033[0m %s\n' "$1" >&2; exit 1; }

if [[ "${EUID}" -ne 0 ]]; then
  die "must be run as root (sudo deploy/bootstrap/install-host.sh)"
fi

# Real (invoking) user, for kubeconfig ownership fixup below.
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
REAL_HOME="$(getent passwd "${REAL_USER}" | cut -d: -f6)"

# ---------------------------------------------------------------------------
# 1. k3s
# ---------------------------------------------------------------------------
log "Step 1/3: k3s (${K3S_VERSION})"

if command -v k3s >/dev/null 2>&1 && k3s --version 2>/dev/null | head -1 | grep -q "${K3S_VERSION#v}"; then
  log "k3s ${K3S_VERSION} already installed, skipping installer."
else
  log "Installing k3s ${K3S_VERSION} (single-node server, traefik disabled — we own ingress explicitly later)."
  curl -sfL https://get.k3s.io | \
    INSTALL_K3S_VERSION="${K3S_VERSION}" \
    INSTALL_K3S_CHANNEL="${K3S_CHANNEL}" \
    INSTALL_K3S_EXEC="server --disable traefik --write-kubeconfig-mode 644" \
    sh -
fi

log "Waiting for k3s node to report Ready..."
for _ in $(seq 1 30); do
  if k3s kubectl get nodes 2>/dev/null | grep -q " Ready"; then
    break
  fi
  sleep 2
done

# Make kubeconfig usable by the invoking (non-root) user without sudo.
mkdir -p "${REAL_HOME}/.kube"
cp /etc/rancher/k3s/k3s.yaml "${REAL_HOME}/.kube/config"
chown "${REAL_USER}:${REAL_USER}" "${REAL_HOME}/.kube/config"
chmod 600 "${REAL_HOME}/.kube/config"
log "Wrote ${REAL_HOME}/.kube/config (owned by ${REAL_USER})."

# ---------------------------------------------------------------------------
# 2. NVIDIA Container Toolkit + k3s containerd runtime wiring
# ---------------------------------------------------------------------------
log "Step 2/3: NVIDIA Container Toolkit"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found — install the NVIDIA driver (not this script's job) before continuing."
fi

if ! dpkg -l 2>/dev/null | grep -q "^ii  nvidia-container-toolkit"; then
  log "Installing nvidia-container-toolkit via the official NVIDIA apt repo."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  # apt has no >= constraint syntax — install latest, then enforce the floor.
  apt-get install -y nvidia-container-toolkit
  _ctk_ver="$(dpkg-query -W -f '${Version}' nvidia-container-toolkit)"
  dpkg --compare-versions "${_ctk_ver}" ge "${NVIDIA_CTK_MIN_VERSION}" || \
    die "nvidia-container-toolkit ${_ctk_ver} < required ${NVIDIA_CTK_MIN_VERSION}"
else
  log "nvidia-container-toolkit already installed, skipping."
fi

# k3s (>= v1.30) AUTO-DETECTS the nvidia runtime: when nvidia-container-runtime
# is on PATH at startup, k3s adds it to its generated containerd config and
# registers the "nvidia" RuntimeClass by itself. Do NOT run `nvidia-ctk runtime
# configure` against k3s's config.toml.tmpl: the template did not previously
# exist, so nvidia-ctk generates a GENERIC containerd config (no k3s CNI
# wiring) and k3s treats the template as a full replacement — the node then
# sticks in NotReady with "cni plugin not initialized". (This exact incident:
# see docs/rearchitecture-v2-decision-log.md, 2026-07-03.)
K3S_CONTAINERD_TMPL="/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl"
if [ -f "${K3S_CONTAINERD_TMPL}" ]; then
  log "Removing stale containerd template so k3s regenerates its own config."
  rm -f "${K3S_CONTAINERD_TMPL}"
fi

log "Restarting k3s so it auto-detects the nvidia runtime."
systemctl restart k3s
for _ in $(seq 1 30); do
  if k3s kubectl get nodes 2>/dev/null | grep -q " Ready"; then
    break
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 3. Sanity checks
# ---------------------------------------------------------------------------
log "Step 3/3: Sanity checks"

echo "--- nvidia-smi ---"
nvidia-smi || warn "nvidia-smi failed — GPU driver problem, not a k3s problem."

echo "--- kubectl get nodes ---"
k3s kubectl get nodes -o wide || die "cluster not responding after install"

echo "--- containerd socket ---"
[[ -S "${K3S_CONTAINERD_SOCK}" ]] && echo "OK: ${K3S_CONTAINERD_SOCK} present" || warn "containerd socket missing at ${K3S_CONTAINERD_SOCK}"

log "Host bootstrap complete."
cat <<EOF

Next steps (run as ${REAL_USER}, no sudo):
  1. deploy/bootstrap/install-user-tools.sh     # kubectl, helm, opentofu -> ~/.local/bin
  2. cd deploy/tofu && tofu init && tofu apply  # GPU Operator, CNPG, Strimzi, SeaweedFS, LGTM
  3. See deploy/README.md for the full runbook and verification/teardown steps.

kubeconfig: ${REAL_HOME}/.kube/config (already exported as your default context)
EOF
