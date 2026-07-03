#!/usr/bin/env bash
# deploy/bootstrap/install-user-tools.sh
#
# No-root installer for the CLI tools used by the rest of deploy/:
# kubectl, helm, opentofu. Installs pinned versions into ~/.local/bin with
# checksum verification against upstream-published SHA256s (embedded below,
# fetched and recorded 2026-07-02 — re-verify if you bump a version).
#
# Run as your normal user, no sudo:
#   deploy/bootstrap/install-user-tools.sh
#
# Idempotent: skips a tool if the pinned version is already installed.

set -euo pipefail

readonly INSTALL_DIR="${HOME}/.local/bin"
readonly TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# ---------------------------------------------------------------------------
# Pinned versions + upstream-published sha256 checksums (linux/amd64 only —
# this repo's target is a single Linux desktop).
# ---------------------------------------------------------------------------
readonly KUBECTL_VERSION="v1.35.6"
readonly KUBECTL_SHA256="5d11e2ba01ea68ffd053f56e27738e2b4330013ee67f7e46c6da6c585d3c9926"

readonly HELM_VERSION="v3.21.2"
readonly HELM_SHA256="0a745198de24545d0055cd8414bc8d2ba10363ef5f5d38369ea1b399671cc083"

readonly TOFU_VERSION="1.12.2"
readonly TOFU_SHA256="2ba8d1aeca94f68aab131628744cd417154462fa5a5b1633b5e04d7618c6ee52"

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
die() { printf '\n\033[1;31mFATAL\033[0m %s\n' "$1" >&2; exit 1; }

verify_sha256() {
  local file="$1" expected="$2"
  local actual
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || die "checksum mismatch for ${file}: expected ${expected}, got ${actual}"
}

mkdir -p "${INSTALL_DIR}"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *) echo "NOTE: ${INSTALL_DIR} is not on your PATH yet. Add to your shell rc:" ;
     echo "  export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

# ---------------------------------------------------------------------------
# kubectl
# ---------------------------------------------------------------------------
log "kubectl ${KUBECTL_VERSION}"
if "${INSTALL_DIR}/kubectl" version --client 2>/dev/null | grep -q "${KUBECTL_VERSION}"; then
  echo "already at ${KUBECTL_VERSION}, skipping."
else
  curl -sfLo "${TMP_DIR}/kubectl" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
  verify_sha256 "${TMP_DIR}/kubectl" "${KUBECTL_SHA256}"
  install -m 0755 "${TMP_DIR}/kubectl" "${INSTALL_DIR}/kubectl"
  echo "installed $(${INSTALL_DIR}/kubectl version --client --output=yaml | grep gitVersion | head -1)"
fi

# ---------------------------------------------------------------------------
# helm
# ---------------------------------------------------------------------------
log "helm ${HELM_VERSION}"
if "${INSTALL_DIR}/helm" version --short 2>/dev/null | grep -q "${HELM_VERSION}"; then
  echo "already at ${HELM_VERSION}, skipping."
else
  curl -sfLo "${TMP_DIR}/helm.tar.gz" "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz"
  verify_sha256 "${TMP_DIR}/helm.tar.gz" "${HELM_SHA256}"
  tar -xzf "${TMP_DIR}/helm.tar.gz" -C "${TMP_DIR}"
  install -m 0755 "${TMP_DIR}/linux-amd64/helm" "${INSTALL_DIR}/helm"
  echo "installed $(${INSTALL_DIR}/helm version --short)"
fi

# ---------------------------------------------------------------------------
# OpenTofu
# ---------------------------------------------------------------------------
log "OpenTofu ${TOFU_VERSION}"
if "${INSTALL_DIR}/tofu" version 2>/dev/null | grep -q "${TOFU_VERSION}"; then
  echo "already at ${TOFU_VERSION}, skipping."
else
  curl -sfLo "${TMP_DIR}/tofu.tar.gz" "https://github.com/opentofu/opentofu/releases/download/v${TOFU_VERSION}/tofu_${TOFU_VERSION}_linux_amd64.tar.gz"
  verify_sha256 "${TMP_DIR}/tofu.tar.gz" "${TOFU_SHA256}"
  tar -xzf "${TMP_DIR}/tofu.tar.gz" -C "${TMP_DIR}"
  install -m 0755 "${TMP_DIR}/tofu" "${INSTALL_DIR}/tofu"
  echo "installed $(${INSTALL_DIR}/tofu version | head -1)"
fi

log "Done. Tools in ${INSTALL_DIR}:"
ls -l "${INSTALL_DIR}/kubectl" "${INSTALL_DIR}/helm" "${INSTALL_DIR}/tofu"
